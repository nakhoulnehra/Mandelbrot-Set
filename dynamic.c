#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex{
  double real;
  double imag;
};

int cal_pixel(struct complex c) {
    double z_real = 0;
    double z_imag = 0;
    double z_real2, z_imag2, lengthsq;
    int iter = 0;
    
    do {
        z_real2 = z_real * z_real;
        z_imag2 = z_imag * z_imag;
        z_imag = 2 * z_real * z_imag + c.imag;
        z_real = z_real2 - z_imag2 + c.real;
        lengthsq = z_real2 + z_imag2;
        iter++;
    }
    while ((iter < MAX_ITER) && (lengthsq < 4.0));
    
    return iter;
}

void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE* pgmimg; 
    int temp;
    pgmimg = fopen(filename, "wb"); 
    fprintf(pgmimg, "P2\n");
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);
    fprintf(pgmimg, "255\n");
    
    for (int i = 0; i < HEIGHT; i++) { 
        for (int j = 0; j < WIDTH; j++) { 
            temp = image[i][j]; 
            fprintf(pgmimg, "%d ", temp);
        } 
        fprintf(pgmimg, "\n"); 
    } 
    fclose(pgmimg); 
}

int main(int argc, char *argv[]) {
    int rank, size;
    int num_trials = 10;
    double average_time = 0;
    double trial_times[num_trials];
    struct complex c;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int (*image)[WIDTH] = malloc(HEIGHT * sizeof(*image));
    int (*local_image)[WIDTH] = malloc(HEIGHT * sizeof(*local_image));

    for (int trial = 0; trial < num_trials; trial++) {
        double start_time, end_time;
        
        for (int i = 0; i < HEIGHT; i++) {
            for (int j = 0; j < WIDTH; j++) {
                local_image[i][j] = 0;
            }
        }
        
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = MPI_Wtime();

        int row = rank;
        while (row < HEIGHT) {
            for (int col = 0; col < WIDTH; col++) {
                c.real = (col - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (row - HEIGHT / 2.0) * 4.0 / HEIGHT;
                local_image[row][col] = cal_pixel(c);
            }
            row += size;
        }

        MPI_Reduce(local_image, image, HEIGHT * WIDTH, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        end_time = MPI_Wtime();

        if (rank == 0) {
            trial_times[trial] = end_time - start_time;
            printf("Execution time of trial [%d]: %f seconds\n", trial + 1, trial_times[trial]);
            average_time += trial_times[trial];
        }
    }

    if (rank == 0) {
        save_pgm("mandelbrot_parallel.pgm", image);
        printf("The average execution time of %d trials is: %f seconds\n", num_trials, average_time / num_trials);
    }

    free(image);
    free(local_image);
    MPI_Finalize();
    return 0;
}