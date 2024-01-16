#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <random>

using namespace Eigen;
using namespace std;
using namespace chrono;

// Function to generate a random positive definite matrix
MatrixXd generatePositiveDefiniteMatrix(int size) {
    MatrixXd A(size, size);
    A.setRandom();

    // Make the matrix symmetric
    A = (A + A.transpose()) / 2.0;

    // Add a multiple of the identity matrix to make it positive definite
    A += size * MatrixXd::Identity(size, size);

    return A;
}

// Function to perform LU decomposition
double performLU(const MatrixXd& A, int iterations) {
    auto start = high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        FullPivLU<MatrixXd> lu(A);
    }

    auto stop = high_resolution_clock::now();
    return duration_cast<milliseconds>(stop - start).count();
}

// Function to perform QR decomposition
double performQR(const MatrixXd& A, int iterations) {
    auto start = high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        HouseholderQR<MatrixXd> qr(A);
    }

    auto stop = high_resolution_clock::now();
    return duration_cast<milliseconds>(stop - start).count();
}

// Function to perform Cholesky decomposition
double performCholesky(const MatrixXd& A, int iterations) {
    auto start = high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        try {
            // Attempt Cholesky decomposition
            LLT<MatrixXd> llt(A);
            if (llt.info() != Eigen::Success) {
                // Cholesky decomposition failed
                cerr << "Cholesky decomposition failed for the given matrix.\n";
                return -1.0;  // Returning a negative value to indicate failure
            }
        } catch (const std::runtime_error& e) {
            // Exception caught if Cholesky decomposition fails
            cerr << "Exception caught: " << e.what() << "\n";
            return -1.0;  // Returning a negative value to indicate failure
        }
    }

    auto stop = high_resolution_clock::now();
    return duration_cast<milliseconds>(stop - start).count();
}

int main() {
    // Set the size of the matrix (adjust as needed)
    const int matrixSize = 60;
    MatrixXd A = generatePositiveDefiniteMatrix(matrixSize);
    const int iterations = 5;

    double qrTime = performQR(A, iterations);
    double luTime = performLU(A, iterations);
    double choleskyTime = performCholesky(A, iterations);

    // Display the time 
    cout << "Matrix Size: " << matrixSize << "x" << matrixSize << endl;
    cout << "Number of Iterations: " << iterations << endl;
    cout << "Average Time taken for QR decomposition: " << qrTime / iterations << " milliseconds\n";
    cout << "Average Time taken for LU decomposition: " << luTime / iterations << " milliseconds\n";
    cout << "Average Time taken for Cholesky decomposition: " << choleskyTime / iterations << " milliseconds\n";

    // Calculate and display the ratios
    cout << "Average Ratios of time taken:\n";
    cout << "QR to LU to Cholesky: " << qrTime / choleskyTime << " : " << luTime / choleskyTime << " : 1.0\n";

    return 0;
}
