/******************************************************************************
 * @file     system_ARMCM55.c
 * @brief    Minimal CMSIS system file for Cortex-M55 on QEMU mps3-an547
 ******************************************************************************/

#if defined(ARMCM55)
#include "ARMCM55.h"
#else
#error device not specified!
#endif

extern const VECTOR_TABLE_Type __VECTOR_TABLE[496];

#define QEMU_MPS3_AN547_SYSTEM_CLOCK (25000000UL)

uint32_t SystemCoreClock = QEMU_MPS3_AN547_SYSTEM_CLOCK;

void SystemCoreClockUpdate(void)
{
    SystemCoreClock = QEMU_MPS3_AN547_SYSTEM_CLOCK;
}

void SystemInit(void)
{
#if defined(__VTOR_PRESENT) && (__VTOR_PRESENT == 1U)
    SCB->VTOR = (uint32_t)(&__VECTOR_TABLE[0]);
#endif

#if defined(__FPU_PRESENT) && (__FPU_PRESENT == 1U)
    /* Enable CP10 and CP11 so Helium and FP instructions are usable from thread mode. */
    SCB->CPACR |= ((3UL << (10U * 2U)) | (3UL << (11U * 2U)));
    __DSB();
    __ISB();
#endif

    SystemCoreClock = QEMU_MPS3_AN547_SYSTEM_CLOCK;
}
