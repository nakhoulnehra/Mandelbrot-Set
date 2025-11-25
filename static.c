#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

#define WIDTH 640
#define HEIGHT 480
#define MAX_ITER 255

struct complex{
  double real;
  double imag;
};

// Function to calculate pixel value for the Mandelbrot set
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

// Function to save the image as a PGM file
void save_pgm(const char *filename, int image[HEIGHT][WIDTH]) {
    FILE* pgmimg; 
    int temp;
    pgmimg = fopen(filename, "wb"); 
    fprintf(pgmimg, "P2\n"); // Writing Magic Number to the File   
    fprintf(pgmimg, "%d %d\n", WIDTH, HEIGHT);  // Writing Width and Height
    fprintf(pgmimg, "255\n");  // Writing the maximum gray value 
    
    for (int i = 0; i < HEIGHT; i++) { 
        for (int j = 0; j < WIDTH; j++) { 
            temp = image[i][j]; 
            fprintf(pgmimg, "%d ", temp); // Writing the gray values in the 2D array to the file 
        } 
        fprintf(pgmimg, "\n"); 
    } 
    fclose(pgmimg); 
}

int main(int argc, char *argv[]) {
    int rank, size;
    int num_trials = 10; // number of trials
    double average_time = 0;
    double trial_times[num_trials];
    struct complex c;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int image[HEIGHT][WIDTH];

    // Loop over the trials
    for (int trial = 0; trial < num_trials; trial++) {
        double start_time, end_time;
        
        MPI_Barrier(MPI_COMM_WORLD); 
        start_time = MPI_Wtime();

        int rows_per_worker = HEIGHT / size;
        int start_row = rank * rows_per_worker;
        int end_row = (rank == size - 1) ? HEIGHT : start_row + rows_per_worker;

        for (int row = start_row; row < end_row; row++) {
            for (int col = 0; col < WIDTH; col++) {
                c.real = (col - WIDTH / 2.0) * 4.0 / WIDTH;
                c.imag = (row - HEIGHT / 2.0) * 4.0 / HEIGHT;
                image[row][col] = cal_pixel(c);
            }
        }
        
        // Send calculated rows to rank 0 
        if (rank != 0) {
            MPI_Send(&image[start_row][0], (end_row - start_row) * WIDTH, MPI_INT, 0, 0, MPI_COMM_WORLD);
        } else {
            for (int worker_rank = 1; worker_rank < size; worker_rank++) {
                int worker_start_row = worker_rank * rows_per_worker;
                int worker_end_row = (worker_rank == size - 1) ? HEIGHT : worker_start_row + rows_per_worker;
                MPI_Recv(&image[worker_start_row][0], (worker_end_row - worker_start_row) * WIDTH, MPI_INT, worker_rank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }

        end_time = MPI_Wtime();

        if (rank == 0) {
            trial_times[trial] = end_time - start_time;
            printf("Execution time of trial [%d]: %f seconds\n", trial + 1, trial_times[trial]);
            average_time += trial_times[trial];
        }
    }

    // Rank 0 saves the image and prints the average execution time
    if (rank == 0) {
        save_pgm("mandelbrot_parallel.pgm", image); 
        printf("The average execution time of %d trials is: %f ms\n", num_trials, (average_time / num_trials) * 1000);
    }

    MPI_Finalize();
    return 0;
}
