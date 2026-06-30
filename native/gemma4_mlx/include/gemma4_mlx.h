#ifndef GEMMA4_MLX_H
#define GEMMA4_MLX_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum Gemma4Status {
    GEMMA4_STATUS_OK = 0,
    GEMMA4_STATUS_UNAVAILABLE = 1
} Gemma4Status;

Gemma4Status gemma4_mlx_smoke(void);

#ifdef __cplusplus
}
#endif

#endif
