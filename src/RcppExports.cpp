// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <Rcpp.h>

using namespace Rcpp;

// fast_group_sum
NumericVector fast_group_sum(NumericVector x, IntegerVector grp);
RcppExport SEXP _taxsimutilities_fast_group_sum(SEXP xSEXP, SEXP grpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type grp(grpSEXP);
    rcpp_result_gen = Rcpp::wrap(fast_group_sum(x, grp));
    return rcpp_result_gen;
END_RCPP
}
// fast_group_max
NumericVector fast_group_max(NumericVector x, IntegerVector grp);
RcppExport SEXP _taxsimutilities_fast_group_max(SEXP xSEXP, SEXP grpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type grp(grpSEXP);
    rcpp_result_gen = Rcpp::wrap(fast_group_max(x, grp));
    return rcpp_result_gen;
END_RCPP
}
// fast_group_min
NumericVector fast_group_min(NumericVector x, IntegerVector grp);
RcppExport SEXP _taxsimutilities_fast_group_min(SEXP xSEXP, SEXP grpSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type grp(grpSEXP);
    rcpp_result_gen = Rcpp::wrap(fast_group_min(x, grp));
    return rcpp_result_gen;
END_RCPP
}
// run_sim_cpp
void run_sim_cpp(int iters, const int M, const int N, const std::vector<int> index, const std::vector<float> U, const std::vector<float> EU, const std::vector<float> V, const std::vector<float> s_v, const std::vector<float> obs_util, const std::vector<float> alt_obs_util, const std::vector<float> h, const std::vector<float> a, const std::vector<float> b, const std::vector<float> sw_cv, const std::vector<float> sw_ev, const std::vector<float> fc, const std::vector<float> inflator, const std::vector<float> sub_disp, const std::vector<float> sub_alt_disp, const std::vector<int> ID, NumericMatrix out, NumericMatrix CV, NumericMatrix EV, NumericMatrix disp_matrix, const std::vector<float> cw, const std::vector<float> lambda, int ncores);
RcppExport SEXP _taxsimutilities_run_sim_cpp(SEXP itersSEXP, SEXP MSEXP, SEXP NSEXP, SEXP indexSEXP, SEXP USEXP, SEXP EUSEXP, SEXP VSEXP, SEXP s_vSEXP, SEXP obs_utilSEXP, SEXP alt_obs_utilSEXP, SEXP hSEXP, SEXP aSEXP, SEXP bSEXP, SEXP sw_cvSEXP, SEXP sw_evSEXP, SEXP fcSEXP, SEXP inflatorSEXP, SEXP sub_dispSEXP, SEXP sub_alt_dispSEXP, SEXP IDSEXP, SEXP outSEXP, SEXP CVSEXP, SEXP EVSEXP, SEXP disp_matrixSEXP, SEXP cwSEXP, SEXP lambdaSEXP, SEXP ncoresSEXP) {
BEGIN_RCPP
    Rcpp::traits::input_parameter< int >::type iters(itersSEXP);
    Rcpp::traits::input_parameter< const int >::type M(MSEXP);
    Rcpp::traits::input_parameter< const int >::type N(NSEXP);
    Rcpp::traits::input_parameter< const std::vector<int> >::type index(indexSEXP);
    Rcpp::traits::input_parameter< const std::vector<float> >::type U(USEXP);
    Rcpp::traits::input_parameter< const std::vector<float> >::type EU(EUSEXP);
    Rcpp::traits::input_parameter< const std::vector<float> >::type V(VSEXP);
    Rcpp::traits::input_parameter< const std::vector<float> >::type s_v(s_vSEXP);
    Rcpp::traits::input_parameter< const std::vector<float> >::type obs_util(obs_utilSEXP);
    Rcpp::traits::input_parameter< const std::vector<float> >::type alt_obs_util(alt_obs_utilSEXP);
    Rcpp::traits::input_parameter< const std::vector<float> >::type h(hSEXP);
    Rcpp::traits::input_parameter< const std::vector<float> >::type a(aSEXP);
    Rcpp::traits::input_parameter< const std::vector<float> >::type b(bSEXP);
    Rcpp::traits::input_parameter< const std::vector<float> >::type sw_cv(sw_cvSEXP);
    Rcpp::traits::input_parameter< const std::vector<float> >::type sw_ev(sw_evSEXP);
    Rcpp::traits::input_parameter< const std::vector<float> >::type fc(fcSEXP);
    Rcpp::traits::input_parameter< const std::vector<float> >::type inflator(inflatorSEXP);
    Rcpp::traits::input_parameter< const std::vector<float> >::type sub_disp(sub_dispSEXP);
    Rcpp::traits::input_parameter< const std::vector<float> >::type sub_alt_disp(sub_alt_dispSEXP);
    Rcpp::traits::input_parameter< const std::vector<int> >::type ID(IDSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type out(outSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type CV(CVSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type EV(EVSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type disp_matrix(disp_matrixSEXP);
    Rcpp::traits::input_parameter< const std::vector<float> >::type cw(cwSEXP);
    Rcpp::traits::input_parameter< const std::vector<float> >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< int >::type ncores(ncoresSEXP);
    run_sim_cpp(iters, M, N, index, U, EU, V, s_v, obs_util, alt_obs_util, h, a, b, sw_cv, sw_ev, fc, inflator, sub_disp, sub_alt_disp, ID, out, CV, EV, disp_matrix, cw, lambda, ncores);
    return R_NilValue;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_taxsimutilities_fast_group_sum", (DL_FUNC) &_taxsimutilities_fast_group_sum, 2},
    {"_taxsimutilities_fast_group_max", (DL_FUNC) &_taxsimutilities_fast_group_max, 2},
    {"_taxsimutilities_fast_group_min", (DL_FUNC) &_taxsimutilities_fast_group_min, 2},
    {"_taxsimutilities_run_sim_cpp", (DL_FUNC) &_taxsimutilities_run_sim_cpp, 27},
    {NULL, NULL, 0}
};

RcppExport void R_init_taxsimutilities(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
