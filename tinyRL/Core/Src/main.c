/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2026 STMicroelectronics.
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
#include "dqn.h"
#include "uart.h"
#include "rng.h"
#include "utils.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
#if TIME_LOG
typedef struct { uint32_t episode, steps, total, forward, backward, adam; } prof_entry_t;
#endif
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#if TIME_LOG
#define PROF_N 50           /* numero di update DQN da profilare prima del dump UART */
#define PROF_DUMP_REPEAT 5  /* ripetizioni del dump: se il link cade durante la
                             * trasmissione il PC ha altre occasioni di riceverlo
                             * (salva solo il primo blocco completo). */
#endif
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
#if TIME_LOG
static prof_entry_t prof_buf[PROF_N];   /* struttura statica: 50 tempi di update */
static uint32_t     prof_count = 0;
static uint8_t      prof_sent  = 0;
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
#define MAX_STEPS_PER_EP  200

/* Topologia della rete Q (e della sua copia target): OBS_DIM -> HIDDEN_DIM ->
 * HIDDEN_DIM -> N_ACTIONS.
 *
 * Sull'H7 l'hidden era 64; qui scende a 32 perche' la NUCLEO-F446RE ha 128 KB
 * di SRAM in tutto contro i 512 KB di RAM_D1 usati sull'H7. Tolti .data+.bss
 * (6.6 KB) e la riserva di stack del linker script (4 KB) restano ~117.4 KB di
 * heap, e con hidden 64 + REPLAY_SIZE 1000 ne servirebbero ~141 KB (rete online
 * 80 KB fra W/dW/mW/vW, target 20.5 KB, replay buffer 40 KB, header di
 * nano-malloc inclusi): init_qnetwork/replay_buffer_init non potrebbero
 * riuscire. Con hidden 32 il conto scende a ~71 KB (24.4 + 6.4 + 40.1) e il
 * replay buffer resta intatto a 1000 transizioni come sull'H7; e' anche la
 * stessa scelta della branch PPO_F446, quindi le curve fra algoritmi restano
 * confrontabili a parita' di rete.
 * Altre configurazioni che entrerebbero, se servisse una rete piu' larga:
 *   HIDDEN_DIM 48, REPLAY_SIZE 1000 -> 101.0 KB (margine 16.4 KB)
 *   HIDDEN_DIM 64, REPLAY_SIZE  400 -> 117.0 KB (margine  0.4 KB, al limite) */
#define HIDDEN_DIM  32

static const float PENDULUM_TORQUES[N_ACTIONS] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};

static float evaluate_reward_pendulum(const float *obs, uint32_t act) {
    float theta = atan2f(obs[1], obs[0]);
    float omega = obs[2];
    float u     = PENDULUM_TORQUES[act];
    return -(theta * theta + 0.1f * omega * omega + 0.001f * u * u);
}

/* Fine dell'update di training: la UART e' rimasta cieca per tutta la durata di
 * dqn_train e nel frattempo il PC ha ritrasmesso l'osservazione, mandando la RX
 * in overrun. Azzeriamo ORE e svuotiamo il registro di ricezione cosi' la
 * lettura successiva riparte allineata su un frame nuovo.
 * Sull'USART della serie F4 non esiste la richiesta di flush della FIFO
 * (l'H7 usa UART_RXDATA_FLUSH_REQUEST): il registro di ricezione e' un solo
 * byte, quindi leggere SR+DR azzera ORE e scarta il byte residuo. */
static void uart_rx_resync(UART_HandleTypeDef *huart) {
    __HAL_UART_CLEAR_OREFLAG(huart);
    __HAL_UART_FLUSH_DRREGISTER(huart);
}

#if TIME_LOG
/* Dump del buffer di profiling come CSV ASCII, racchiuso tra i marcatori che
 * il parser lato PC cerca. Formato:
 *   <<<PROF_BEGIN>>>header\n riga\n ... <<<PROF_END>>>
 * (nessun \n subito dopo BEGIN, cosi' block.count("\n")-1 == numero di righe). */
static void prof_dump_csv(UART_HandleTypeDef *huart)
{
  char line[96];
  const char *begin = "<<<PROF_BEGIN>>>";
  const char *end   = "<<<PROF_END>>>";

  HAL_UART_Transmit(huart, (uint8_t *)begin, strlen(begin), 1000);

  int n = snprintf(line, sizeof(line),
                   "episode,steps,total_ms,forward_ms,backward_ms,adam_ms\n");
  HAL_UART_Transmit(huart, (uint8_t *)line, n, 1000);

  for (uint32_t i = 0; i < prof_count; i++) {
    /* cicli -> microsecondi interi, poi stampati come ms con 3 decimali
       (us/1000 . us%1000): niente printf-float, nessun flag di linker extra. */
    uint32_t t = cycles_to_us(prof_buf[i].total);
    uint32_t f = cycles_to_us(prof_buf[i].forward);
    uint32_t b = cycles_to_us(prof_buf[i].backward);
    uint32_t a = cycles_to_us(prof_buf[i].adam);
    n = snprintf(line, sizeof(line),
                 "%lu,%lu,%lu.%03lu,%lu.%03lu,%lu.%03lu,%lu.%03lu\n",
                 (unsigned long)prof_buf[i].episode,
                 (unsigned long)prof_buf[i].steps,
                 (unsigned long)(t / 1000), (unsigned long)(t % 1000),
                 (unsigned long)(f / 1000), (unsigned long)(f % 1000),
                 (unsigned long)(b / 1000), (unsigned long)(b % 1000),
                 (unsigned long)(a / 1000), (unsigned long)(a % 1000));
    HAL_UART_Transmit(huart, (uint8_t *)line, n, 1000);
  }
  HAL_UART_Transmit(huart, (uint8_t *)end, strlen(end), 1000);
}
#endif /* TIME_LOG */
/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void) {

  /* USER CODE BEGIN 1 */
  /* Il Cortex-M4 della F446 non ha cache L1 ne' MPU da configurare: la
   * MPU_Config() del build H7 sparisce. */
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick.
   */
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

  QNetwork      online;
  TargetNetwork target;
  ReplayBuffer  replay;

  int topology[]        = {OBS_DIM, HIDDEN_DIM, HIDDEN_DIM, N_ACTIONS};
  ActivationType acts[] = {ACT_RELU, ACT_RELU, ACT_NONE};

  int init_ok = init_qnetwork(&online, sizeof(topology) / sizeof(topology[0]),
                             topology, acts) &&
                init_target_network(&target,
                                    sizeof(topology) / sizeof(topology[0]),
                                    topology) &&
                replay_buffer_init(&replay, REPLAY_SIZE, OBS_DIM);

  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
  if (init_ok) {
      copy_weights_to_target(&online, &target);
  } else {
      /* Heap esaurito: la topologia/replay richiesti non entrano nei 128 KB di
       * SRAM (vedi il commento su HIDDEN_DIM). La NUCLEO-F446RE ha un solo LED
       * utente (LD2, PA5): lo facciamo lampeggiare veloce - distinguibile dal
       * LED acceso fisso durante gli update - e diciamo sulla seriale cos'e'
       * andato storto, cosi' l'errore non e' muto. */
      char err[64];
      int  n = snprintf(err, sizeof(err),
                        "<<<ERR>>>ALLOC hidden=%d replay=%d\n",
                        (int)HIDDEN_DIM, (int)REPLAY_SIZE);
      while (1) {
          HAL_UART_Transmit(&huart2, (uint8_t *)err, n, 1000);
          for (int i = 0; i < 5; i++) {
              HAL_GPIO_TogglePin(LD2_GPIO_Port, LD2_Pin);
              HAL_Delay(100);
          }
      }
  }

  float    obs[OBS_DIM];
  float    prev_obs[OBS_DIM];
  uint32_t action      = 0;
  uint32_t step_total  = 0;
  uint32_t train_step  = 0;
  uint32_t num_episode = 0;
  uint32_t step_in_ep  = 0;
  uint8_t  first_step  = 1;
  uint8_t  manual_done;
  float    manual_reward;
  float    epsilon;
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1) {
    if (!uart_recv_floats(&huart2, obs, OBS_DIM, 100))
        continue;

    int obs_ok = 1;
    for (int i = 0; i < OBS_DIM; i++)
        if (!isfinite(obs[i]) || fabsf(obs[i]) > 1000.0f) { obs_ok = 0; break; }
    if (!obs_ok)
        continue;   // frame sporco: il PC lo ritrasmette

    manual_done   = (step_in_ep >= MAX_STEPS_PER_EP);
    manual_reward = first_step ? 0.0f
                               : evaluate_reward_pendulum(obs, action);

    if (!first_step)
        replay_buffer_push(&replay, prev_obs, action,
                           manual_reward, obs, manual_done);

    if (replay.size >= REPLAY_MIN) {
        HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_SET);
        dqn_train(&online, &target, &replay, BATCH_SIZE, N_ACTIONS);
#if TIME_LOG
        /* --- profiling: i tempi (total/forward/backward/adam) del singolo
         *     update sono in g_train_timing, misurati dentro dqn_train --- */
        if (prof_count < PROF_N) {
            prof_buf[prof_count].episode  = num_episode;
            prof_buf[prof_count].steps    = train_step;
            prof_buf[prof_count].total    = g_train_timing.total_cycles;
            prof_buf[prof_count].forward  = g_train_timing.forward_cycles;
            prof_buf[prof_count].backward = g_train_timing.backward_cycles;
            prof_buf[prof_count].adam     = g_train_timing.adam_cycles;
            prof_count++;
        }
#endif
        train_step++;
        if (train_step % TARGET_UPDATE == 0)
            copy_weights_to_target(&online, &target);
        HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);
        /* A 180 MHz l'update dura comunque piu' che sull'H7 a 480 MHz: la RX e'
         * sicuramente in overrun, va risincronizzata prima di rispondere. */
        uart_rx_resync(&huart2);
    }

    epsilon = calc_epsilon(step_total);
    action  = dqn_select_action(&online, obs, epsilon, N_ACTIONS);

    memcpy(prev_obs, obs, OBS_DIM * sizeof(float));
    first_step = 0;

    uart_send_float_action(&huart2, PENDULUM_TORQUES[action], manual_done, 100);

    if (manual_done) {
        memset(prev_obs, 0, sizeof(prev_obs));
        first_step = 1;
        step_in_ep = 0;
        num_episode++;
#if TIME_LOG
        /* buffer pieno: invia il CSV dei tempi a fine episodio (boundary pulito
         * per il parser lato PC), ripetendolo per qualche episodio in caso di
         * caduta del link. */
        if (prof_count == PROF_N && prof_sent < PROF_DUMP_REPEAT) {
            prof_dump_csv(&huart2);
            prof_sent++;
            uart_rx_resync(&huart2);   /* il dump e' lungo: RX di nuovo in overrun */
        }
#endif
    } else {
        step_in_ep++;
    }
    step_total++;
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
  }
  /* USER CODE END 3 */
}

/**
 * @brief System Clock Configuration
 * @retval None
 */
void SystemClock_Config(void) {
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage
   */
  /* VOS1 (scale 1) e' richiesto per salire oltre i 144 MHz: gli altri build
   * F446, fermi a 84 MHz, usavano VOS3. */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  /** Initializes the RCC Oscillators according to the specified parameters
   * in the RCC_OscInitTypeDef structure.
   */
  /* HSI 16 MHz -> PLL: M=8 (2 MHz all'ingresso del VCO), N=180 (VCO 360 MHz),
   * P=2 -> 180 MHz SYSCLK, il massimo della F446 (l'H7 girava a 480 MHz).
   * NB: gli altri build F446 (REINFORCE su main, PPO su PPO_F446) girano a
   * 84 MHz, quindi i tempi di training di questa branch NON sono direttamente
   * confrontabili con i loro: per confrontarli va applicata la stessa
   * configurazione di clock anche la'. */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 8;
  RCC_OscInitStruct.PLL.PLLN = 180;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Activate the Over-Drive mode
   */
  /* Obbligatoria sopra i 168 MHz e da abilitare proprio qui: il PLL e' gia'
   * configurato ma il SYSCLK non ci e' ancora sopra. */
  if (HAL_PWREx_EnableOverDrive() != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
   */
  /* HCLK = 180 MHz; i bus periferici hanno un tetto proprio, PCLK1 45 MHz e
   * PCLK2 90 MHz, quindi i divisori salgono a 4 e 2. */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  /* 5 wait state: a 2.7-3.6 V la flash regge 30 MHz per WS, quindi 180 MHz ne
   * richiede 5 (a 84 MHz bastavano 2). Prefetch e ART accelerator sono attivi
   * da stm32f4xx_hal_conf.h, cosi' la latenza pesa poco sul codice lineare. */
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
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
void Error_Handler(void) {
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1) {
  }
  /* USER CODE END Error_Handler_Debug */
}
#ifdef USE_FULL_ASSERT
/**
 * @brief  Reports the name of the source file and the source line number
 *         where the assert_param error has occurred.
 * @param  file: pointer to the source file name
 * @param  line: assert_param error line source number
 * @retval None
 */
void assert_failed(uint8_t *file, uint32_t line) {
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line
     number, ex: printf("Wrong parameters value: file %s on line %d\r\n", file,
     line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
