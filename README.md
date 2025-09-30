# STM-RL

**STM-RL** is an On-Device Reinforcement Learning framework designed for STM32 boards.

## Overview

STM-RL provides a lightweight reinforcement learning (RL) implementation that runs directly on STM32 microcontrollers. It leverages STM32 HAL and CMSIS drivers for hardware abstraction and is suited for embedded RL experimentation, prototyping, and deployment in resource-constrained environments.

## Features

- On-device RL algorithms for STM32 platforms
- Neural network and dense layer implementation in C
- Utilizes STM32 HAL and CMSIS for hardware interfacing
- Designed for real-time, low-power, resource-limited applications
- Modular and extensible codebase for custom RL agents

## Directory Structure

```
tinyRL/
├── Core/              # Main source code and user logic
│   ├── Src/           # RL agent, neural net, dense layer sources
│   ├── Inc/           # Header files
│   └── Startup/       # Startup code for STM32
├── Drivers/
│   ├── CMSIS/         # ARM CMSIS drivers and headers
│   └── STM32F4xx_HAL_Driver/ # STM32F4 HAL drivers
├── Debug/             # Build artifacts (auto-generated)
```

## Getting Started

1. **Requirements**
    - STM32 development board (e.g., STM32F446RE)
    - STM32CubeIDE or compatible toolchain
    - ARM GCC toolchain

2. **Build & Flash**
    - Open the project in STM32CubeIDE.
    - Build the project; binaries are produced in `Debug/`.
    - Flash the compiled binary (`tinyRL.elf`) onto your STM32 board.

3. **Customize RL Agent**
    - Modify the RL logic in `Core/Src/reinforce.c` and neural network code in `Core/Src/neural_net.c`.
    - Adjust hardware-specific code in `Core/Src/main.c` as needed.

## License

This project contains components under the [Apache-2.0 License](https://opensource.org/licenses/Apache-2.0), including CMSIS and STM32 HAL drivers. See the `Drivers/CMSIS/LICENSE.txt` and `Drivers/CMSIS/Device/ST/STM32F4xx/LICENSE.txt` for details.

## Acknowledgements

- [ARM CMSIS](https://github.com/ARM-software/CMSIS_5)
- [STMicroelectronics STM32 HAL](https://www.st.com/en/embedded-software/stm32cube-mcu-packages.html)

---

*Developed by [kartman88](https://github.com/kartman88)*
