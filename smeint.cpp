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

#if 0
typedef std::complex<double> dtype;
typedef Eigen::SparseMatrix<dtype> SpMat;
typedef Eigen::MatrixXcd DMat;
#else
typedef double dtype;
typedef Eigen::SparseMatrix<dtype> SpMat;
typedef Eigen::MatrixXd DMat;
typedef Eigen::RowVectorXd RVec;
#endif

struct SMESolver {
    SMESolver(
            SpMat hamiltonian
    ) : hamiltonian(hamiltonian) {}

    enum TermCond {
        kGreaterThanOrEqualTo,
        kLessThan,
        kStepLimitReached
    };

    SpMat hamiltonian;
    DMat rho0;

    std::vector<SpMat> collapse;

    // Observed operator and the efficiency it comes in with
    std::vector<SpMat> measurement;
    std::vector<double> etas;

    // All expectation values of interest
    std::vector<SpMat> ev_ops;

    // Key from names of expectation values to their index
    std::unordered_map<std::string, int> ev_names;

    // Termination condition (expval index, condition, value)
    std::vector<std::tuple<int, TermCond, dtype>> term_conds;

    void AddCollapse(SpMat op);
    void AddMeasurement(SpMat op, double eta);

    void AddExpectationValue(std::string name, SpMat op);
    void AddTerminationCondition(std::string name, TermCond tc, dtype value);

    std::tuple<DMat,DMat> Run(DMat rho0, int N, double dt);
};

void SMESolver::AddExpectationValue(std::string name, SpMat op){
    ev_ops.push_back(op);
    ev_names[name] = ev_ops.size() - 1;
}

void SMESolver::AddTerminationCondition(
        std::string name,
        SMESolver::TermCond tc,
        dtype value){

    int idx = ev_names[name];
    term_conds.push_back(std::tie(idx, tc, value));
}

void SMESolver::AddCollapse(SpMat op){
    collapse.push_back(op);
}

void SMESolver::AddMeasurement(SpMat op, double eta){
    measurement.push_back(op);
    etas.push_back(eta);
}

std::tuple<DMat,DMat> SMESolver::Run(DMat rho0, int N, double dt){
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
        throw new
           std::domain_error("rho0 shape doesn't match system dimension");
    }

    for( auto &v : collapse )
    if(v.rows() != D || v.cols() != D) {
        std::cout<<"collapse ops"<<std::endl;
        throw new std::domain_error("Collapse operator has wrong dimension");
    }

    for( auto &l : measurement )
    if(l.rows() != D || l.cols() != D) {
        std::cout<<"msmt ops"<<std::endl;
        throw new
            std::domain_error("Measurement operator has wrong dimension");
    }

    DMat dy(N,M);
    DMat rhos(N,D*D);
    DMat rho(rho0);

    // Deterministic part of the evolution
    //    SpMat mdet = dtype{0.0,1.0} * hamiltonian;
    SpMat mdet = 0.0 * hamiltonian;
    for( auto &v : collapse )    mdet += 0.5*v.adjoint()*v;
    for( auto &l : measurement ) mdet += 0.5*l.adjoint()*l;

    for(int i=0;i<N;i++){
        // Measurement record
        for(int r=0;r<M;r++) {
            dtype a = (measurement[r] * rho \
                    + rho * (measurement[r].adjoint())).trace();

            dy(i,r) = sqrt(etas[r]) * a * dt + d(gen)*rtdt;
        }

        DMat m = DMat::Identity(D,D) - mdet*dt;

        for(int r=0;r<M;r++){
            m += (sqrt(etas[r])*dy(i,r))*measurement[r];

            for(int s=0;s<M;s++){
                dtype f = dy(i,r)*dy(i,s);
                if(r == s) f -= dt;

                m += 0.5*sqrt(etas[r]*etas[s])*measurement[r]*measurement[s]*f;
            }
        }

        DMat rhonext = m*rho*m.adjoint();
        for( auto &v : collapse ) rhonext += v*rho*v.adjoint()*dt;

        for(int j=0;j<measurement.size();j++)
            rhonext += \
                (1-etas[j])*measurement[j]*rho*measurement[j].adjoint()*dt;

        rhonext /= rhonext.trace();

        Eigen::Map<RVec> rhocol(rhonext.data(),rhonext.size());

        rhos.row(i) = rhocol;
        rho = rhonext;
    }

    return std::tie(dy,rhos);
}

PYBIND11_PLUGIN(smeint) {
    py::module m("smeint", "SME Solver");

    py::class_<SMESolver> solver(m, "Solver");

    solver.def(py::init<SpMat>())
          .def("add_collapse",              &SMESolver::AddCollapse)
          .def("add_measurement",           &SMESolver::AddMeasurement)
          .def("add_expectation_value",     &SMESolver::AddExpectationValue)
          .def("add_termination_condition", &SMESolver::AddTerminationCondition)
          .def("run",                       &SMESolver::Run);

    py::enum_<SMESolver::TermCond>(solver, "TermCond")
        .value("LessThan",          SMESolver::TermCond::kLessThan)
        .value("LessThanOrEqualTo", SMESolver::TermCond::kGreaterThanOrEqualTo);

    return m.ptr();
}
