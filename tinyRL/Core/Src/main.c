/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "neural_net.h"
#include "uart.h"
#include "reinforce.h"
#include "utils.h"
#include "rng.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
#if TIME_LOG
static uint32_t num_episode = 0;   /* episodi conclusi (done inviato al PC) */
static uint32_t train_step  = 0;   /* update REINFORCE eseguiti */
#endif
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

/* ── Task: CartPole-v1 ────────────────────────────────────────────────────────
 * obs = [x, x_dot, theta, theta_dot], azioni discrete {0 = sinistra, 1 = destra}.
 * Terminazione: carrello oltre +-2.4 m, asta oltre +-12 gradi, o 500 step. */
#define CART_LIMIT  2.4f          /* +-2.4 m  */
#define POLE_LIMIT  0.20943951f   /* +-12 deg */

#if !USE_CONTINUOUS_ACTION
static float compute_reward(const float *obs, uint32_t act) {
    (void)obs; (void)act;
    return 1.0f;   // CartPole: +1 per ogni step non terminale
}
#else
static float compute_reward(const float *obs, const float *action) {
    (void)obs; (void)action;
    return 1.0f;
}
#endif

// Osservazione corrente condivisa con is_done (aggiornata nel loop prima di
// reinforce_step): la terminazione di CartPole dipende dallo stato, non solo
// dal contatore di step.
static float s_cur_obs[OBS_DIM];

static uint8_t is_done(uint32_t step_in_ep) {
    if (step_in_ep >= MAX_STEPS_PER_EP)   return 1;   // troncamento a 500 step
    if (fabsf(s_cur_obs[0]) > CART_LIMIT) return 1;   // fuori pista
    if (fabsf(s_cur_obs[2]) > POLE_LIMIT) return 1;   // asta caduta
    return 0;
}

static void on_train_begin(void) {
    // La NUCLEO-F446RE ha un solo LED utente (LD2, PA5): lo usiamo come
    // indicatore di training, acceso per tutta la durata dell'update.
    HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);
}

#if TIME_LOG
/* Una riga CSV di timing per ogni update, spedita subito dopo l'update stesso.
 * Il PC la appende al file mentre il training prosegue: niente buffer da
 * riempire e nessun dump unico da perdere, ogni riga e' indipendente. Formato:
 *   <<<PROF>>>episode,steps,total_ms,forward_ms,backward_ms,adam_ms\n
 * I tempi arrivano da g_train_timing (cicli a 64 bit) e vengono convertiti in
 * microsecondi, poi stampati come ms con 3 decimali (us/1000 . us%1000): niente
 * printf-float e niente long long, che newlib-nano non sempre supporta. */
static void prof_emit_row(UART_HandleTypeDef *huart)
{
  char line[128];
  uint32_t t = cycles64_to_us(g_train_timing.total_cycles);
  uint32_t f = cycles64_to_us(g_train_timing.forward_cycles);
  uint32_t b = cycles64_to_us(g_train_timing.backward_cycles);
  uint32_t a = cycles64_to_us(g_train_timing.adam_cycles);

  int n = snprintf(line, sizeof(line),
                   "<<<PROF>>>%lu,%lu,%lu.%03lu,%lu.%03lu,%lu.%03lu,%lu.%03lu\n",
                   (unsigned long)num_episode,
                   (unsigned long)train_step,
                   (unsigned long)(t / 1000), (unsigned long)(t % 1000),
                   (unsigned long)(f / 1000), (unsigned long)(f % 1000),
                   (unsigned long)(b / 1000), (unsigned long)(b % 1000),
                   (unsigned long)(a / 1000), (unsigned long)(a % 1000));
  HAL_UART_Transmit(huart, (uint8_t *)line, n, 1000);
}
#endif /* TIME_LOG */

static void on_train_end(void) {
    HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

#if TIME_LOG
    /* I tempi (total/forward/backward/adam) di questo update sono in
     * g_train_timing, misurati dentro reinforce_update: spediamo la riga adesso,
     * prima del flush RX qui sotto, cosi' il PC la legge nello stesso momento in
     * cui torna a leggere l'azione. */
    prof_emit_row(&huart2);
    train_step++;
#endif

    // Durante la pausa di training la UART e' andata in overrun e nella RX si
    // sono accumulati byte (ritrasmissioni del PC). Azzeriamo l'overrun e
    // svuotiamo la RX cosi' la lettura riparte allineata su un frame nuovo.
    // Sull'USART della serie F4 non esiste la richiesta di flush della FIFO
    // (l'H7 usa UART_RXDATA_FLUSH_REQUEST): il registro di ricezione e' un solo
    // byte, quindi leggere SR+DR azzera ORE e scarta il byte residuo.
    __HAL_UART_CLEAR_OREFLAG(&huart2);
    __HAL_UART_FLUSH_DRREGISTER(&huart2);
}
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();
  /* USER CODE BEGIN 2 */
  dwt_init();
  rng_seed(HAL_GetTick());

  Network        policy;
  ReinforceAgent agent;

#if USE_CONTINUOUS_ACTION
  int topology[]        = {OBS_DIM, 64, 64, REINFORCE_POLICY_OUT_DIM};
  ActivationType acts[] = {ACT_RELU, ACT_RELU, ACT_NONE};
#else
  int topology[]        = {OBS_DIM, 64, REINFORCE_POLICY_OUT_DIM};
  ActivationType acts[] = {ACT_RELU, ACT_SOFTMAX};
#endif

  int init_ok = network_init(&policy, sizeof(topology) / sizeof(topology[0]),
                             topology, acts) &&
                reinforce_agent_init(&agent, &policy,
                                     compute_reward, is_done,
                                     on_train_begin, on_train_end);

  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
  if (!init_ok) {
      // Allocazione fallita: con un solo LED utente segnaliamo l'errore con un
      // lampeggio veloce, cosi' resta distinguibile dal LED acceso a fisso
      // durante gli update di training.
      while (1) {
          HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);
          HAL_Delay(100);
      }
  }

  float obs[OBS_DIM];
#if USE_CONTINUOUS_ACTION
  float action[N_ACT_DIMS];
#endif
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    if (!uart_recv_floats(&huart2, obs, OBS_DIM, 100))
        continue;

    // Ultima difesa (oltre a checksum+ETX in uart_recv_floats): scarta i frame
    // con osservazioni non finite o di magnitudine assurda. Un frame disallineato
    // post-overrun puo' contenere float spazzatura finiti ma enormi (es. 1e30):
    // passerebbero isfinite, ma un obs gigante avvelenerebbe i ritorni
    // Monte-Carlo. La soglia 1000 e' molto sopra i valori legit.
    int obs_ok = 1;
    for (int i = 0; i < OBS_DIM; i++)
        if (!isfinite(obs[i]) || fabsf(obs[i]) > 1000.0f) { obs_ok = 0; break; }
    if (!obs_ok)
        continue;

    memcpy(s_cur_obs, obs, OBS_DIM * sizeof(float));   // letto da is_done()

#if USE_CONTINUOUS_ACTION
    reinforce_step(&agent, obs, action);
    for (int i = 0; i < N_ACT_DIMS; i++) {
        if (action[i] >  ACTION_SCALE) action[i] =  ACTION_SCALE;
        if (action[i] < -ACTION_SCALE) action[i] = -ACTION_SCALE;
    }
    uart_send_floats_action(&huart2, action, N_ACT_DIMS, agent.done, 100);
#else
    uint32_t act = reinforce_step_discrete(&agent, obs);
    uart_send_action_discrete(&huart2, act, agent.done, 100);
#endif

#if TIME_LOG
    if (agent.done)
        num_episode++;   /* colonna "episode" delle righe di timing */
#endif

    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
  */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{

  /* USER CODE BEGIN USART2_Init 0 */

  /* USER CODE END USART2_Init 0 */

  /* USER CODE BEGIN USART2_Init 1 */

  /* USER CODE END USART2_Init 1 */
  huart2.Instance = USART2;
  huart2.Init.BaudRate = 115200;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits = UART_STOPBITS_1;
  huart2.Init.Parity = UART_PARITY_NONE;
  huart2.Init.Mode = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART2_Init 2 */

  /* USER CODE END USART2_Init 2 */

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
/* USER CODE BEGIN MX_GPIO_Init_1 */
/* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : LD2_Pin */
  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);

/* USER CODE BEGIN MX_GPIO_Init_2 */
/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
