#include <sstream>                    // String to number conversion
#include <windows.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h> 
#include <time.h>

#define EPOCHFILETIME   (116444736000000000UL)

//GPUスレッドでの実行関数
__global__ void cudaAdd(double *array_a, double *array_b, double *array_c, int arraySize)
{
	//スレッドの番号を計算
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if (tid < arraySize) {
		//スレッド番号を配列のインデックスとして、
		//配列aと配列bのこのインデックスの値の加算して、配列cに入れる。
		array_c[tid] = array_a[tid] + array_b[tid];
	}
}

//現在の時刻を取得
int64_t getCurrentTime()
{
	FILETIME ft;
	LARGE_INTEGER li;
	int64_t tt = 0;
	GetSystemTimeAsFileTime(&ft);
	li.LowPart = ft.dwLowDateTime;
	li.HighPart = ft.dwHighDateTime;
	tt = (li.QuadPart - EPOCHFILETIME) / 10 / 10;
	return tt;
}

//CPUで計算
void runCPU(double *array_a, double *array_b, double *array_c, int arraySize)
{
	//ループで加算する
	for (int i = 0; i < arraySize; i++) {
		array_c[i] = array_a[i] + array_b[i];
	}
}

//GPUで計算
extern "C" void runGPU(double *array_a, double *array_b, double *array_g, int arraySize) {
	int size = arraySize * sizeof(double);

	//GPUメモリのアロケーション
	double *array_a_g;
	double *array_b_g;
	double *array_c_g;
	cudaMalloc(&array_a_g, size);
	cudaMalloc(&array_b_g, size);
	cudaMalloc(&array_c_g, size);

	//データをCPUメモリからGPUメモリへコピー
	cudaMemcpy(array_a_g, array_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(array_b_g, array_b, size, cudaMemcpyHostToDevice);

	//グリッドとスレッドブロックを当てる
	dim3 blockSize(16);
	dim3 gridSize((arraySize + blockSize.x - 1) / blockSize.x);

	//GPU関数を呼び出す
	cudaAdd << <gridSize, blockSize >> >(array_a_g, array_b_g, array_c_g, arraySize);

	//GPU計算を待つ
	cudaThreadSynchronize();

	//データをGPUメモリからCPUメモリへコピー
	cudaMemcpy(array_g, array_c_g, size, cudaMemcpyDeviceToHost);

	//GPU領域解放
	cudaFree(array_a_g);
	cudaFree(array_b_g);
	cudaFree(array_c_g);
}


int main1(int, char *argv[])
{
	//GPU初期化
	cudaFree(nullptr);

	int N = 100000000;
	int size = N * sizeof(double);
	printf("arraySize:%d\n",N);

	//CPUメモリのアロケーション
	double *array_a = new double[N];
	double *array_b = new double[N];
	double *array_c = new double[N];
	double *array_g = new double[N];

	//ランダムデータを作成
	srand(time(NULL));
	for (int i = 0; i < N; i++) {
		array_a[i] = (i + 1) * 1.2 * rand() / RAND_MAX;
		array_b[i] = (i + 1) * 1.2 * rand() / RAND_MAX;
	}

	//CPUで計算
	int64_t csT = getCurrentTime();
	runCPU(array_a, array_b, array_c, N);
	csT = getCurrentTime() - csT;
	printf("CPUTime:%d\n",csT);

	//GPUで計算
	int64_t gsT = getCurrentTime();
	runGPU(array_a, array_b, array_g, N);
	gsT = getCurrentTime() - gsT;
	printf("GPUTime:%d\n", gsT);

	//結果をチェック
	for (int i = 0; i < N; i++) {
		if (array_c[i] != array_g[i]) {
			printf("has error!\n");
			printf("cpu[%d]:%f\n", i, array_c[i]);
			printf("gpu[%d]:%f\n", i, array_g[i]);
			break;
		}
	}

	//CPU領域解放
	free(array_a);
	free(array_b);
	free(array_c);
	free(array_g);

	system("pause");
	return 0;
}