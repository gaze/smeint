#include <iostream>
#include <tuple>
#include <vector>
#include <Eigen/Core>
#include <memory>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <complex>
#include <cmath>
#include <assert.h>
#include <random>

namespace py = pybind11;

struct SMESolver {
    SMESolver(
            Eigen::MatrixXcd hamiltonian
    ) : hamiltonian(hamiltonian) {}

    Eigen::MatrixXcd hamiltonian;
    Eigen::MatrixXcd rho0;

    std::vector<Eigen::MatrixXcd> collapse;

    // Observed operator and the efficiency it comes in with
    std::vector<Eigen::MatrixXcd> measurement;
    std::vector<double> etas;

    void AddCollapse(Eigen::MatrixXcd op);
    void AddMeasurement(Eigen::MatrixXcd op, double eta);

    std::tuple<Eigen::MatrixXcd,Eigen::MatrixXcd> Run(Eigen::MatrixXd rho0, int N, double dt);
};

void SMESolver::AddCollapse(Eigen::MatrixXcd op){
    collapse.push_back(op);
}

void SMESolver::AddMeasurement(Eigen::MatrixXcd op, double eta){
    measurement.push_back(op);
    etas.push_back(eta);
}

std::tuple<Eigen::MatrixXcd,Eigen::MatrixXcd> SMESolver::Run(Eigen::MatrixXd rho0, int N, double dt){
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> d;

    double rtdt = sqrt(dt);

    unsigned long M = measurement.size();

    // Domain checks
    if(hamiltonian.rows() != hamiltonian.cols()){
        std::cout<<"Hamiltonian not square"<<std::endl;
        throw new std::domain_error("Hamiltonian not square");
    }
    long D = hamiltonian.rows();

    if(rho0.rows() != D || rho0.cols() != D){
        std::cout<<"rho0"<<std::endl;
        throw new std::domain_error("rho0 shape doesn't match system dimension");
    }

    for( auto &v : collapse )
        if(v.rows() != D || v.cols() != D) {
            std::cout<<"collapse ops"<<std::endl;
            throw new std::domain_error("Collapse operator has wrong dimension");
        }

    for( auto &l : measurement )
        if(l.rows() != D || l.cols() != D) {
            std::cout<<"msmt ops"<<std::endl;
            throw new std::domain_error("Measurement operator has wrong dimension");
        }

    Eigen::MatrixXcd dy(N,M);
    Eigen::MatrixXcd rhos(N,D*D);
    Eigen::MatrixXcd rho(rho0);

    // Deterministic part of the evolution
    Eigen::MatrixXcd mdet = std::complex<double>{0.0,1.0} * hamiltonian;
    for( auto &v : collapse )    mdet += 0.5*v.adjoint()*v;
    for( auto &l : measurement ) mdet += 0.5*l.adjoint()*l;

    for(int i=0;i<N;i++){

        // Measurement record
        for(int r=0;r<M;r++) {
            std::complex<double> a = (measurement[r] * rho + rho * (measurement[r].adjoint())).trace();
            dy(i,r) = sqrt(etas[r]) * a * dt + d(gen)*rtdt;
        }

        Eigen::MatrixXcd m = Eigen::MatrixXcd::Identity(D,D) - mdet*dt;

        for(int r=0;r<M;r++){
            m += (sqrt(etas[r])*dy(i,r))*measurement[r];

            for(int s=0;s<M;s++){
                std::complex<double> f = dy(i,r)*dy(i,s);
                if(r == s) f -= dt;

                m += 0.5*sqrt(etas[r]*etas[s])*measurement[r]*measurement[s]*f;
            }
        }

        Eigen::MatrixXcd rhonext = m*rho*m.adjoint();
        for( auto &v : collapse ) rhonext += v*rho*v.adjoint()*dt;

        for(int j=0;j<measurement.size();j++)
            rhonext += (1-etas[j])*measurement[j]*rho*measurement[j].adjoint()*dt;

        rhonext /= rhonext.trace();

        Eigen::Map<Eigen::RowVectorXcd> rhocol(rhonext.data(),rhonext.size());

        rhos.row(i) = rhocol;
        rho = rhonext;
    }

    return std::tie(dy,rhos);
}

int main() {
    std::cout << "Hello, World!" << std::endl;
    return 0;
}

PYBIND11_PLUGIN(smeint) {
    py::module m("smeint", "SME Solver");

    py::class_<SMESolver>(m, "Solver")
            .def(py::init<Eigen::MatrixXcd>())
            .def("add_collapse", &SMESolver::AddCollapse)
            .def("add_measurement", &SMESolver::AddMeasurement)
            .def("run", &SMESolver::Run);

    return m.ptr();
}
