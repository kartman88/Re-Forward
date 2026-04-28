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
#include "reinforce.h"
#include "utils.h"
#include <stdio.h>

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

UART_HandleTypeDef huart3;

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MPU_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART3_UART_Init(void);
/* USER CODE BEGIN PFP */
uint8_t done_check_swimmer(uint32_t ep_step);
float evaluate_reward_swimmer(float *obs, action_t *prev_actions);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

uint8_t done_check_swimmer(uint32_t ep_step) {
  // Swimmer non termina mai, solo truncation a 1000 step
  if (ep_step >= ROLLOUT) {
    return 1;
  }
  return 0;
}

float evaluate_reward_swimmer(float *obs, action_t *prev_actions) {
  // obs[3] = velocità x della punta (front tip) ≈ forward_reward (dx/dt)
  float forward_reward = obs[3];
  // ctrl_cost = 0.0001 * ||actions||²
  float ctrl_cost = 0.0001f * (prev_actions[0] * prev_actions[0] +
                               prev_actions[1] * prev_actions[1]);
  return forward_reward - ctrl_cost;
}

/* USER CODE END 0 */

/**
 * @brief  The application entry point.
 * @retval int
 */
int main(void) {

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* MPU Configuration--------------------------------------------------------*/
  MPU_Config();

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
  MX_USART3_UART_Init();
  /* USER CODE BEGIN 2 */
  dwt_init();
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  // random seed to generate initial weight
  srand(HAL_GetTick());
  //-----NETWORK PARAMETERS AND CREATION-----
  int input_size = 8; // Swimmer: 3 angoli + 5 velocità

#if USE_CONTINUOUS_ACTIONS
  int output_size = 2; // Swimmer: 2 torque (rotori)
  ActivationType activations_actor_layers[] = {
      ACT_TANH}; // Bound action space inside [-1, 1] range!
#else
  int output_size = 2; // N azioni discrete
  ActivationType activations_actor_layers[] = {ACT_RELU, ACT_SOFTMAX};
#endif

  // create neural network
  SharedBackbone net;
  int num_layers = 2;        // Backbone: 8 -> 64
  int num_layers_actor = 2;  // Actor: 32 -> 2
  int num_layers_critic = 2; // Critic: 32 -> 1
  int net_topology[] = {input_size, 64};
  int net_topology_actor[] = {32, output_size};
  int net_topology_critic[] = {32, 1};
  ActivationType activations[] = {ACT_RELU, ACT_RELU}; // Per i Trunk Layers
  ActivationType activations_critic_layers[] = {
      ACT_NONE}; // Layer interni Critic

  int is_ok = init_network(&net, num_layers, num_layers_actor,
                           num_layers_critic, net_topology, net_topology_actor,
                           net_topology_critic, activations,
                           activations_actor_layers, activations_critic_layers);

  //-----BUFFER PARAMETER AND CREATION-----
  Buffer buffer;
  int buffer_size = MAX_STEPS;
  buffer_init(&buffer, buffer_size, input_size, output_size);
  uint32_t step_count = 0;
  uint32_t num_episode = 0;
  uint8_t manual_done = 0;
  action_t action[output_size]; // Array di N azioni
  float manual_reward = 0;
  float obs[input_size];
  uint8_t train = 1;

  // Variabili Locali per la logica dell'ambiente (ex-Pendulum)
  action_t prev_action_raw[output_size]; // Array di N azioni precedenti
  for (int i = 0; i < output_size; i++)
    prev_action_raw[i] = 0.0f;
  uint32_t ep_step = 0;

  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_RESET);
  HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_RESET);
  if (is_ok)
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_SET);
  else
    HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_SET);

  while (1) {
    if (uart_recv_floats(&huart3, obs, net_topology[0], 50)) {
      HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_RESET);
      HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_0);

      // --- ENVIRONMENT LOGIC (SWIMMER) ---
      // 1. Calcolo del DONE (Solo timeout)
      manual_done = done_check_swimmer(ep_step);

      // 2. Calcolo del REWARD (forward velocity - ctrl cost)
      manual_reward = evaluate_reward_swimmer(obs, prev_action_raw);

      // --- CALLING GENERIC RL ENGINE ---
      int status = step(&net, obs, manual_reward, manual_done, action,
                        &step_count, &buffer);

      // Salviamo l'azione generata per il calcolo del reward al prossimo giro
      for (int i = 0; i < output_size; i++)
        prev_action_raw[i] = action[i];

      // --- 1. SE IL BUFFER È PIENO -> TRAINING ---
      if (status == 2) {
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_0, GPIO_PIN_RESET);
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_SET);
        if (train == 1)
          finish_episode(&buffer, &net, MAX_STEPS,
                         1); // Pass 1 for buffer-full termination
        HAL_GPIO_WritePin(GPIOB, GPIO_PIN_14, GPIO_PIN_RESET);
        step_count = 0;
        ep_step = 0; // Reset local counter along with buffer
        // num_episode lo incrementiamo sotto, quando inviamo il done
      }

      // --- 2. COMUNICAZIONE UART CON L'AMBIENTE ---
      ep_step++; // Increment step tracker

      // Se il gioco continua (manual_done == 0), invia le azioni direttamente
      // (Swimmer usa [-1, 1] nativo, nessuno scaling necessario)
      if (manual_done == 0) {
        uart_send_action(&huart3, action, output_size, manual_done, 50);
      }
      // Se l'episodio è finito, invia il reset e azzera
      else {
        float zero_actions[output_size];
        for (int i = 0; i < output_size; i++)
          zero_actions[i] = 0.0f;
        uart_send_action(&huart3, zero_actions, output_size, manual_done, 50);
        num_episode++;
        ep_step = 0;
        for (int i = 0; i < output_size; i++)
          prev_action_raw[i] = 0.0f;
      }
    }

    if (num_episode > MAX_EPISODE) {
      // train = 0;
      // break;
    }
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

  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /* VOS1 richiesto per 480 MHz */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);
  while (!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /* HSI 64 MHz → PLL1: DIVM=4 (16 MHz), DIVN=60 (960 MHz VCO), DIVP=2 → 480 MHz SYSCLK */
  RCC_OscInitStruct.OscillatorType      = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState            = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState        = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource       = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM            = 4;
  RCC_OscInitStruct.PLL.PLLN            = 60;
  RCC_OscInitStruct.PLL.PLLP            = 2;   /* SYSCLK = 480 MHz */
  RCC_OscInitStruct.PLL.PLLQ            = 4;   /* 240 MHz, disponibile per periferiche */
  RCC_OscInitStruct.PLL.PLLR            = 2;
  RCC_OscInitStruct.PLL.PLLRGE          = RCC_PLL1VCIRANGE_3; /* VCI 8-16 MHz */
  RCC_OscInitStruct.PLL.PLLVCOSEL       = RCC_PLL1VCOWIDE;    /* VCO 192-960 MHz */
  RCC_OscInitStruct.PLL.PLLFRACN        = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK) {
    Error_Handler();
  }

  /* SYSCLK = PLL1P = 480 MHz
     AHB = 240 MHz (DIV2), APB1/2/3/4 = 120 MHz (DIV2) */
  RCC_ClkInitStruct.ClockType      = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK |
                                     RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2 |
                                     RCC_CLOCKTYPE_D3PCLK1 | RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource   = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider  = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider  = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;

  /* Flash latency 4 cicli richiesti a 480 MHz VOS1 */
  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK) {
    Error_Handler();
  }
}

/**
 * @brief USART3 Initialization Function
 * @param None
 * @retval None
 */
static void MX_USART3_UART_Init(void) {

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 115200;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart3.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart3) != HAL_OK) {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart3, UART_TXFIFO_THRESHOLD_1_8) !=
      HAL_OK) {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart3, UART_RXFIFO_THRESHOLD_1_8) !=
      HAL_OK) {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart3) != HAL_OK) {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */
}

/**
 * @brief GPIO Initialization Function
 * @param None
 * @retval None
 */
static void MX_GPIO_Init(void) {
  GPIO_InitTypeDef GPIO_InitStruct = {0};
  /* USER CODE BEGIN MX_GPIO_Init_1 */

  /* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOE_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(GPIOB, LD1_Pin | LD3_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  /*Configure GPIO pin : B1_Pin */
  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_INPUT;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : LD1_Pin LD3_Pin */
  GPIO_InitStruct.Pin = LD1_Pin | LD3_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

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

/* MPU Configuration */

void MPU_Config(void) {
  MPU_Region_InitTypeDef MPU_InitStruct = {0};

  /* Disables the MPU */
  HAL_MPU_Disable();

  /** Initializes and configures the Region and the memory to be protected
   */
  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER0;
  MPU_InitStruct.BaseAddress = 0x0;
  MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
  MPU_InitStruct.SubRegionDisable = 0x87;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;

  HAL_MPU_ConfigRegion(&MPU_InitStruct);
  /* Enables the MPU */
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);
}

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
