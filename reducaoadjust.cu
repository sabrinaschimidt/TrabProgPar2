// Operação de redução (soma) dos elementos de um vetor
// Usa vários blocos de threads (quando necessário)
// Alunas: Sabrina Renata Gonçalves Schimidt
// Cláudia Magno Pereira de Brito
// Melhorias implmentadas de 1 a 6
// Para compilar: nvcc reducaoadjust.cu -o reducao
// Para executar: ./reducao número_de_elementos


#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define MAX_THREADS_BLOCO 1024  // Tamanho máximo do bloco (definido pela GPU)

// Redução 
double reducao_CPU(int n, double *a)
{
    double s = 0;
    for (int i = 0; i < n; i++)
    {
        s = s + a[i];
    }
    return s;
}

// Kernel executado na GPU por todas as threads de todos os blocos
__global__ void reducao_GPU(int n, double *a)
{
    extern __shared__ double sharedData[];

    int tIdLocal = threadIdx.x;                                      // id LOCAL da thread
    int tIdGlobal = blockIdx.x * blockDim.x + threadIdx.x;           // id GLOBAL da thread

    // Carrega os dados na memória compartilhada
    sharedData[tIdLocal] = (tIdGlobal < n) ? a[tIdGlobal] : 0.0;
    __syncthreads();

    // Faz log_2(blockDim.x) níveis da redução na memória compartilhada
    for (int nThAtivas = blockDim.x / 2; nThAtivas > 0; nThAtivas = nThAtivas >> 1)
    {
        if (tIdLocal < nThAtivas)
            sharedData[tIdLocal] += sharedData[tIdLocal + nThAtivas];

        // Sincronização de barreira entre threads do bloco
        __syncthreads();
    }

    // Thread 0 escreve resultado na memória global
    if (tIdLocal == 0)
        a[blockIdx.x] = sharedData[0];
}

// Programa principal: execução inicia no host
int main(int argc, char **argv)
{
    int n,                    // Número de elementos do vetor de entrada
        nBytes,               // Número de bytes do vetor de entrada
        nBlocos,              // Tamanho do grid
        nThreadsBloco;        // Tamanho do bloco
    double *hVet,             // Vetor de entrada do host
        *dVet,                // Vetor de entrada da GPU
        hResult,              // Resultado calculado no host
        dResult;              // Resultado calculado na GPU

    if (argc != 2)
    {
        printf("O programa foi executado com parâmetros incorretos.\n");
        printf("Uso: ./reducao número_de_elementos\n");
        exit(1);
    }

    n = atoi(argv[1]);

    nBytes = n * sizeof(double);

    // Aloca vetor de entrada no host
    hVet = (double *)malloc(nBytes);

    // Inicializa vetor de entrada do host
    for (int i = 0; i < n; i++)
        hVet[i] = (double)i;

    struct timeval h_ini, h_fim;
    gettimeofday(&h_ini, 0);

    // Calcula redução na CPU
    hResult = reducao_CPU(n, hVet);

    gettimeofday(&h_fim, 0);
    long segundos = h_fim.tv_sec - h_ini.tv_sec;
    long microsegundos = h_fim.tv_usec - h_ini.tv_usec;
    double h_tempo = (segundos * 1e3) + (microsegundos * 1e-3); // Tempo de execução na CPU em ms

    // Aloca vetor de entrada na memória global da GPU
    cudaMalloc((void **)&dVet, nBytes);

    cudaEvent_t d_ini, d_fim;
    cudaEventCreate(&d_ini);
    cudaEventCreate(&d_fim);
    cudaEventRecord(d_ini, 0);

    // Copia vetor de entrada do host para memória global da GPU
    cudaMemcpy(dVet, hVet, nBytes, cudaMemcpyHostToDevice);

    // Calcula redução na GPU
    nThreadsBloco = MAX_THREADS_BLOCO;
    nBlocos = (n + nThreadsBloco - 1) / nThreadsBloco;

    while (nBlocos > 1)
    {
        reducao_GPU<<<nBlocos, nThreadsBloco, nThreadsBloco * sizeof(double)>>>(n, dVet);

        // Atualiza configuração de blocos e threads para a próxima redução
        n = nBlocos;
        nThreadsBloco = MAX_THREADS_BLOCO;
        nBlocos = (n + nThreadsBloco - 1) / nThreadsBloco;
    }

    // Copia resultado da memória global da GPU para host
    cudaMemcpy(&dResult, dVet, sizeof(double), cudaMemcpyDeviceToHost);

    cudaEventRecord(d_fim, 0);
    cudaEventSynchronize(d_fim);
    float d_tempo; // Tempo de execução na GPU em ms
    cudaEventElapsedTime(&d_tempo, d_ini, d_fim);
    cudaEventDestroy(d_ini);
    cudaEventDestroy(d_fim);

    printf("Tempo CPU = %.2fms\t Tempo GPU = %.2fms\n", h_tempo, d_tempo);

    // Checa resultado
    printf("%s\n", ((dResult != hResult) ? "Resultado ERRADO" : "Resultado correto"));

    // Libera vetor de entrada na memória global da GPU
    cudaFree(dVet);

    // Libera vetor de entrada no host
    free(hVet);

    return 0;
}
