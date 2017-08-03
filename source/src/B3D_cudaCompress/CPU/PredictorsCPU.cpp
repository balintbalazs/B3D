#include <cudaCompress/CPU/PredictorsCPU.h>
#include <math.h>


namespace cudaCompress {

	namespace util {

		void predictor7_tilesCPU(const int16_t* in, int16_t* out, int pitch, int width, int height, int tileSize)
		{

			int p = pitch / sizeof(int16_t);
			int tilesX = (width + tileSize - 1) / tileSize;
			int tilesY = (height + tileSize - 1) / tileSize;
			
			//#pragma omp parallel for
			for (int ty = 0; ty < tilesY; ty++)
			{
				for (int tx = 0; tx < tilesX; tx++)
				{
					for (int v = 0; v < tileSize; v++)
					{
						for (int u = 0; u < tileSize; u++)
						{
							int x = u + tx * tileSize;
							int y = v + ty * tileSize;

							if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height) {
								if (u == 0) {
									if (v > 0) {
										out[p*y + x] = in[p*y + x] - in[p*(y - 1) + x];
									}
									else {
										out[p*y + x] = in[p*y + x];
									}
								}
								else if (v == 0) {
									out[p*y + x] = in[p*y + x] - in[p*y + x - 1];
								}
								else {
									out[p*y + x] = in[p*y + x] - (uint16_t)((int32_t)(in[p*y + x - 1] + (int32_t)in[p*(y - 1) + x]) >> 1);
								}
							}

						}
					}
				}
			}

			
			return;
		}

		void unPredictor7_tilesCPU(const int16_t* in, int16_t* out, int pitch, int width, int height, int tileSize)
		{
			int p = pitch / sizeof(int16_t);
			int tilesX = (width + tileSize - 1) / tileSize;
			int tilesY = (height + tileSize - 1) / tileSize;
			
			//#pragma omp parallel for
			for (int ty = 0; ty < tilesY; ty++)
			{
				for (int tx = 0; tx < tilesX; tx++)
				{
					for (int v = 0; v < tileSize; v++)
					{
						for (int u = 0; u < tileSize; u++)
						{
							int x = u + tx * tileSize;
							int y = v + ty * tileSize;

							if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height) {
								if (u == 0) {
									if (v > 0) {
										out[p*y + x] = in[p*y + x] + out[p*(y - 1) + x];
									}
									else {
										out[p*y + x] = in[p*y + x];
									}
								}
								else if (v == 0) {
									out[p*y + x] = in[p*y + x] + out[p*y + x - 1];
								}
								else {
									out[p*y + x] = in[p*y + x] + (uint16_t)((int32_t)(out[p*y + x - 1] + (int32_t)out[p*(y - 1) + x]) >> 1);
								}
							}

						}
					}
				}
			}
			return;
		}

		void predictor7_tiles_wnllCPU(const float* in, float* buffer, int16_t* out, int pitch, int width, int height, int tileSize)
		{
			int p = pitch / sizeof(int16_t);
			int tilesX = (width + tileSize - 1) / tileSize;
			int tilesY = (height + tileSize - 1) / tileSize;

			for (int ty = 0; ty < tilesY; ty++)
			{
				for (int tx = 0; tx < tilesX; tx++)
				{
					for (int v = 0; v < tileSize; v++)
					{
						for (int u = 0; u < tileSize; u++)
						{
							int x = u + tx * tileSize;
							int y = v + ty * tileSize;

							if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height) {
								if (u == 0) {
									if (v > 0) {
										float eps = in[p*y + x] - buffer[p*(y - 1) + x];
										int Q = eps + (eps < 0 ? -0.5 : 0.5);
										out[p*y + x] = Q;
										buffer[p*y + x] = Q + buffer[p*(y - 1) + x];
									}
									else {
										float eps = in[p*y + x];
										int Q = eps + (eps < 0 ? -0.5 : 0.5);
										out[p*y + x] = Q;
										buffer[p*y + x] = Q;
									}
								}
								else if (v == 0) {
									float eps = in[p*y + x] - buffer[p*y + x - 1];
									int Q = eps + (eps < 0 ? -0.5 : 0.5);
									out[p*y + x] = Q;
									buffer[p*y + x] = Q + buffer[p*y + x - 1];
								}
								else {
									float eps = in[p*y + x] - ((buffer[p*y + x - 1] + buffer[p*(y - 1) + x]) / 2);
									int Q = eps + (eps < 0 ? -0.5 : 0.5);
									//int Q = eps + 0.5;
									out[p*y + x] = Q;
									buffer[p*y + x] = Q + ((buffer[p*y + x - 1] + buffer[p*(y - 1) + x]) / 2);
								}
							}

						}
					}
				}
			}
			return;
		}

		void unPredictor7_tiles_wnllCPU(const int16_t* in, float* out, int pitch, int width, int height, int tileSize)
		{
			int p = pitch / sizeof(int16_t);
			int tilesX = (width + tileSize - 1) / tileSize;
			int tilesY = (height + tileSize - 1) / tileSize;

			for (int ty = 0; ty < tilesY; ty++)
			{
				for (int tx = 0; tx < tilesX; tx++)
				{
					for (int v = 0; v < tileSize; v++)
					{
						for (int u = 0; u < tileSize; u++)
						{
							int x = u + tx * tileSize;
							int y = v + ty * tileSize;

							if (u >= 0 && v >= 0 && u < tileSize && v < tileSize && x < width && y < height) {
								if (u == 0) {
									if (v > 0) {
										out[p*y + x] = in[p*y + x] + out[p*(y - 1) + x];
									}
									else {
										out[p*y + x] = in[p*y + x];
									}
								}
								else if (v == 0) {
									out[p*y + x] = in[p*y + x] + out[p*y + x - 1];
								}
								else {
									out[p*y + x] = in[p*y + x] + ((out[p*y + x - 1] + out[p*(y - 1) + x]) / 2);
								}
							}

						}
					}
				}
			}
			return;
		}


		void vstCPU(float* in, float* out, int num, float offset, float conversion, float sigma)
		{
			//#pragma omp parallel for
			for (int x = 0; x < num; x++)
			{
				out[x] = 2 * sqrtf((fmaxf(in[x] - offset, 0)) / conversion + sigma*sigma) - 2 * sigma;
			}
			return;
		}

		void invVstCPU(float* in, float* out, int num, float offset, float conversion, float sigma)
		{
			float D = 0;
			//#pragma omp parallel for
			for (int x = 0; x < num; x++)
			{
				D = in[x];
				D = D + 2 * sigma; // remove offset
				if (D >= 2 * sigma) {
					out[x] = ((D*D / 4) - sigma*sigma)*conversion + offset;
				}
				else {
					out[x] = offset;
				}
			}
			return;
		}


		void s2fCPU(int16_t* in, float* out, int num)
		{
			//#pragma omp parallel for
			for (int x = 0; x < num; x++)
			{
				out[x] = in[x];
			}
			return;
		}

		void f2sCPU(float* in, int16_t* out, int num)
		{
			//#pragma omp parallel for
			for (int x = 0; x < num; x++)
			{
				out[x] = in[x] + (in[x] < 0 ? -1 : 1) * 0.5;
			}
			return;
		}

		void u2fCPU(uint16_t* in, float* out, int num)
		{
			//#pragma omp parallel for
			for (int x = 0; x < num; x++)
			{
				out[x] = in[x];
			}
			return;
		}

		void f2uCPU(float* in, uint16_t* out, int num)
		{
			//#pragma omp parallel for
			for (int x = 0; x < num; x++)
			{
				out[x] = in[x] + (in[x] < 0 ? -1 : 1) * 0.5;
			}
			return;
		}


		void multiplyCPU(float* in, float* out, float factor, int num)
		{
			//#pragma omp parallel for
			for (int x = 0; x < num; x++)
			{
				out[x] = in[x] * factor;
			}
			return;
		}

		void multiplyCPU(int16_t* in, float* out, float factor, int num)
		{
			//#pragma omp parallel for
			for (int x = 0; x < num; x++)
			{
				out[x] = in[x] * factor;
			}
			return;
		}

	}

}