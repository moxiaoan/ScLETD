# Project Name

Phase-field models have been used extensively in studying microstructure evolution in alloys and have the superiority of comprehending, predicting, 
and optimizing microstructure-sensitive macroscopic material properties. ScLETD focuses on supporting the simulations of solid-solid phase transformation.


## Installation

To install the project, follow these steps:

1. Clone the repository: https://github.com/moxiaoan/ScLETD
2. Install dependencies and load the environment:
   compiler/devtoolset/7.3.1<br>
   mpi/hpcx/2.11.0/gcc-7.3.1
   compiler/rocm/2.9
4. Build the software:
   cd ScLETD_based_large-scale_simulation
   make clean && make -j 20
5. Run the executable file: ./run.sh nodes 192

## Contributing

Contributions are always welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch for your changes.
3. Make your changes and commit them.
4. Open a pull request.

## Contact

For any questions or suggestions, feel free to reach out to me:

- Email: [gaoyaqian@cnic.cn]
