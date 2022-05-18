// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

#ifdef RCPP_USE_GLOBAL_ROSTREAM
Rcpp::Rostream<true>&  Rcpp::Rcout = Rcpp::Rcpp_cout_get();
Rcpp::Rostream<false>& Rcpp::Rcerr = Rcpp::Rcpp_cerr_get();
#endif

// greg_cpp_one
arma::mat greg_cpp_one(arma::colvec W, arma::mat& C, arma::vec& B);
RcppExport SEXP _taxsimutilities_greg_cpp_one(SEXP WSEXP, SEXP CSEXP, SEXP BSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type W(WSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type C(CSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type B(BSEXP);
    rcpp_result_gen = Rcpp::wrap(greg_cpp_one(W, C, B));
    return rcpp_result_gen;
END_RCPP
}
// fast_bs_sum
NumericVector fast_bs_sum(NumericVector pattern);
RcppExport SEXP _taxsimutilities_fast_bs_sum(SEXP patternSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type pattern(patternSEXP);
    rcpp_result_gen = Rcpp::wrap(fast_bs_sum(pattern));
    return rcpp_result_gen;
END_RCPP
}
// logSumExp
double logSumExp(NumericVector& logV, int accumulators);
RcppExport SEXP _taxsimutilities_logSumExp(SEXP logVSEXP, SEXP accumulatorsSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector& >::type logV(logVSEXP);
    Rcpp::traits::input_parameter< int >::type accumulators(accumulatorsSEXP);
    rcpp_result_gen = Rcpp::wrap(logSumExp(logV, accumulators));
    return rcpp_result_gen;
END_RCPP
}
// stable_ig
NumericVector stable_ig(const NumericMatrix U, const NumericMatrix W, const int N, const int M);
RcppExport SEXP _taxsimutilities_stable_ig(SEXP USEXP, SEXP WSEXP, SEXP NSEXP, SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< const NumericMatrix >::type U(USEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type W(WSEXP);
    Rcpp::traits::input_parameter< const int >::type N(NSEXP);
    Rcpp::traits::input_parameter< const int >::type M(MSEXP);
    rcpp_result_gen = Rcpp::wrap(stable_ig(U, W, N, M));
    return rcpp_result_gen;
END_RCPP
}
// stable_n
NumericVector stable_n(const NumericMatrix U, const NumericMatrix W, const int N, const int M);
RcppExport SEXP _taxsimutilities_stable_n(SEXP USEXP, SEXP WSEXP, SEXP NSEXP, SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< const NumericMatrix >::type U(USEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type W(WSEXP);
    Rcpp::traits::input_parameter< const int >::type N(NSEXP);
    Rcpp::traits::input_parameter< const int >::type M(MSEXP);
    rcpp_result_gen = Rcpp::wrap(stable_n(U, W, N, M));
    return rcpp_result_gen;
END_RCPP
}
// stable_p
NumericVector stable_p(const NumericMatrix U, const NumericMatrix W, const int N, const int M);
RcppExport SEXP _taxsimutilities_stable_p(SEXP USEXP, SEXP WSEXP, SEXP NSEXP, SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< const NumericMatrix >::type U(USEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type W(WSEXP);
    Rcpp::traits::input_parameter< const int >::type N(NSEXP);
    Rcpp::traits::input_parameter< const int >::type M(MSEXP);
    rcpp_result_gen = Rcpp::wrap(stable_p(U, W, N, M));
    return rcpp_result_gen;
END_RCPP
}
// llc_cpp
double llc_cpp(const NumericVector p, const NumericMatrix H1, const NumericMatrix H2, const NumericMatrix H1sq, const NumericMatrix H2sq, const NumericMatrix H1H2, const NumericMatrix Y, const NumericMatrix H1Y, const NumericMatrix H2Y, const NumericMatrix Ysq, const NumericMatrix TW, const NumericVector nw, const int& N, const int& M);
RcppExport SEXP _taxsimutilities_llc_cpp(SEXP pSEXP, SEXP H1SEXP, SEXP H2SEXP, SEXP H1sqSEXP, SEXP H2sqSEXP, SEXP H1H2SEXP, SEXP YSEXP, SEXP H1YSEXP, SEXP H2YSEXP, SEXP YsqSEXP, SEXP TWSEXP, SEXP nwSEXP, SEXP NSEXP, SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< const NumericVector >::type p(pSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type H1(H1SEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type H2(H2SEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type H1sq(H1sqSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type H2sq(H2sqSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type H1H2(H1H2SEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type H1Y(H1YSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type H2Y(H2YSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type Ysq(YsqSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type TW(TWSEXP);
    Rcpp::traits::input_parameter< const NumericVector >::type nw(nwSEXP);
    Rcpp::traits::input_parameter< const int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< const int& >::type M(MSEXP);
    rcpp_result_gen = Rcpp::wrap(llc_cpp(p, H1, H2, H1sq, H2sq, H1H2, Y, H1Y, H2Y, Ysq, TW, nw, N, M));
    return rcpp_result_gen;
END_RCPP
}
// llc_alt_cpp
double llc_alt_cpp(const SEXP p, const SEXP H1, const SEXP H2, const SEXP H1sq, const SEXP H2sq, const SEXP H1H2, const SEXP Y, const SEXP H1Y, const SEXP H2Y, const SEXP Ysq, const SEXP lTW, const SEXP nw, const int& N, const int& M);
RcppExport SEXP _taxsimutilities_llc_alt_cpp(SEXP pSEXP, SEXP H1SEXP, SEXP H2SEXP, SEXP H1sqSEXP, SEXP H2sqSEXP, SEXP H1H2SEXP, SEXP YSEXP, SEXP H1YSEXP, SEXP H2YSEXP, SEXP YsqSEXP, SEXP lTWSEXP, SEXP nwSEXP, SEXP NSEXP, SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< const SEXP >::type p(pSEXP);
    Rcpp::traits::input_parameter< const SEXP >::type H1(H1SEXP);
    Rcpp::traits::input_parameter< const SEXP >::type H2(H2SEXP);
    Rcpp::traits::input_parameter< const SEXP >::type H1sq(H1sqSEXP);
    Rcpp::traits::input_parameter< const SEXP >::type H2sq(H2sqSEXP);
    Rcpp::traits::input_parameter< const SEXP >::type H1H2(H1H2SEXP);
    Rcpp::traits::input_parameter< const SEXP >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const SEXP >::type H1Y(H1YSEXP);
    Rcpp::traits::input_parameter< const SEXP >::type H2Y(H2YSEXP);
    Rcpp::traits::input_parameter< const SEXP >::type Ysq(YsqSEXP);
    Rcpp::traits::input_parameter< const SEXP >::type lTW(lTWSEXP);
    Rcpp::traits::input_parameter< const SEXP >::type nw(nwSEXP);
    Rcpp::traits::input_parameter< const int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< const int& >::type M(MSEXP);
    rcpp_result_gen = Rcpp::wrap(llc_alt_cpp(p, H1, H2, H1sq, H2sq, H1H2, Y, H1Y, H2Y, Ysq, lTW, nw, N, M));
    return rcpp_result_gen;
END_RCPP
}
// lls_cpp
double lls_cpp(const NumericVector p, const NumericMatrix H, const NumericMatrix Hsq, const NumericMatrix Y, const NumericMatrix Ysq, const NumericMatrix HY, const NumericMatrix TW, const NumericVector nw, const int& N, const int& M);
RcppExport SEXP _taxsimutilities_lls_cpp(SEXP pSEXP, SEXP HSEXP, SEXP HsqSEXP, SEXP YSEXP, SEXP YsqSEXP, SEXP HYSEXP, SEXP TWSEXP, SEXP nwSEXP, SEXP NSEXP, SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< const NumericVector >::type p(pSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type H(HSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type Hsq(HsqSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type Ysq(YsqSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type HY(HYSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type TW(TWSEXP);
    Rcpp::traits::input_parameter< const NumericVector >::type nw(nwSEXP);
    Rcpp::traits::input_parameter< const int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< const int& >::type M(MSEXP);
    rcpp_result_gen = Rcpp::wrap(lls_cpp(p, H, Hsq, Y, Ysq, HY, TW, nw, N, M));
    return rcpp_result_gen;
END_RCPP
}
// lls_alt_cpp
double lls_alt_cpp(const SEXP p, const SEXP H, const SEXP Hsq, const SEXP Y, const SEXP Ysq, const SEXP HY, const SEXP lTW, const SEXP nw, const int& N, const int& M);
RcppExport SEXP _taxsimutilities_lls_alt_cpp(SEXP pSEXP, SEXP HSEXP, SEXP HsqSEXP, SEXP YSEXP, SEXP YsqSEXP, SEXP HYSEXP, SEXP lTWSEXP, SEXP nwSEXP, SEXP NSEXP, SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< const SEXP >::type p(pSEXP);
    Rcpp::traits::input_parameter< const SEXP >::type H(HSEXP);
    Rcpp::traits::input_parameter< const SEXP >::type Hsq(HsqSEXP);
    Rcpp::traits::input_parameter< const SEXP >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const SEXP >::type Ysq(YsqSEXP);
    Rcpp::traits::input_parameter< const SEXP >::type HY(HYSEXP);
    Rcpp::traits::input_parameter< const SEXP >::type lTW(lTWSEXP);
    Rcpp::traits::input_parameter< const SEXP >::type nw(nwSEXP);
    Rcpp::traits::input_parameter< const int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< const int& >::type M(MSEXP);
    rcpp_result_gen = Rcpp::wrap(lls_alt_cpp(p, H, Hsq, Y, Ysq, HY, lTW, nw, N, M));
    return rcpp_result_gen;
END_RCPP
}
// llsopt_cpp
double llsopt_cpp(const NumericVector p, const NumericMatrix H, const NumericMatrix Hsq, const NumericMatrix Y, const NumericMatrix Ysq, const NumericMatrix HY, const NumericMatrix TW, const NumericVector nw, const int& N, const int& M, const int& opt_mode);
RcppExport SEXP _taxsimutilities_llsopt_cpp(SEXP pSEXP, SEXP HSEXP, SEXP HsqSEXP, SEXP YSEXP, SEXP YsqSEXP, SEXP HYSEXP, SEXP TWSEXP, SEXP nwSEXP, SEXP NSEXP, SEXP MSEXP, SEXP opt_modeSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< const NumericVector >::type p(pSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type H(HSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type Hsq(HsqSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type Y(YSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type Ysq(YsqSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type HY(HYSEXP);
    Rcpp::traits::input_parameter< const NumericMatrix >::type TW(TWSEXP);
    Rcpp::traits::input_parameter< const NumericVector >::type nw(nwSEXP);
    Rcpp::traits::input_parameter< const int& >::type N(NSEXP);
    Rcpp::traits::input_parameter< const int& >::type M(MSEXP);
    Rcpp::traits::input_parameter< const int& >::type opt_mode(opt_modeSEXP);
    rcpp_result_gen = Rcpp::wrap(llsopt_cpp(p, H, Hsq, Y, Ysq, HY, TW, nw, N, M, opt_mode));
    return rcpp_result_gen;
END_RCPP
}
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
// gini_sorted
double gini_sorted(const std::vector<double> y, const std::vector<double> w);
RcppExport SEXP _taxsimutilities_gini_sorted(SEXP ySEXP, SEXP wSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double> >::type y(ySEXP);
    Rcpp::traits::input_parameter< const std::vector<double> >::type w(wSEXP);
    rcpp_result_gen = Rcpp::wrap(gini_sorted(y, w));
    return rcpp_result_gen;
END_RCPP
}
// fast_poverty
NumericMatrix fast_poverty(const std::vector<double> y, const std::vector<double> w, const std::vector<double> k, int ncores);
RcppExport SEXP _taxsimutilities_fast_poverty(SEXP ySEXP, SEXP wSEXP, SEXP kSEXP, SEXP ncoresSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< const std::vector<double> >::type y(ySEXP);
    Rcpp::traits::input_parameter< const std::vector<double> >::type w(wSEXP);
    Rcpp::traits::input_parameter< const std::vector<double> >::type k(kSEXP);
    Rcpp::traits::input_parameter< int >::type ncores(ncoresSEXP);
    rcpp_result_gen = Rcpp::wrap(fast_poverty(y, w, k, ncores));
    return rcpp_result_gen;
END_RCPP
}
// fast_med
double fast_med(NumericVector y, NumericVector w);
RcppExport SEXP _taxsimutilities_fast_med(SEXP ySEXP, SEXP wSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector >::type w(wSEXP);
    rcpp_result_gen = Rcpp::wrap(fast_med(y, w));
    return rcpp_result_gen;
END_RCPP
}
// fast_pov
double fast_pov(NumericVector y, NumericVector w, double thres);
RcppExport SEXP _taxsimutilities_fast_pov(SEXP ySEXP, SEXP wSEXP, SEXP thresSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector >::type w(wSEXP);
    Rcpp::traits::input_parameter< double >::type thres(thresSEXP);
    rcpp_result_gen = Rcpp::wrap(fast_pov(y, w, thres));
    return rcpp_result_gen;
END_RCPP
}
// fast_cpov
double fast_cpov(NumericVector y, NumericVector w, NumericVector k, double thres);
RcppExport SEXP _taxsimutilities_fast_cpov(SEXP ySEXP, SEXP wSEXP, SEXP kSEXP, SEXP thresSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector >::type w(wSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type k(kSEXP);
    Rcpp::traits::input_parameter< double >::type thres(thresSEXP);
    rcpp_result_gen = Rcpp::wrap(fast_cpov(y, w, k, thres));
    return rcpp_result_gen;
END_RCPP
}
// fast_povgap
double fast_povgap(NumericVector y, NumericVector w, double thres, double exp);
RcppExport SEXP _taxsimutilities_fast_povgap(SEXP ySEXP, SEXP wSEXP, SEXP thresSEXP, SEXP expSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector >::type w(wSEXP);
    Rcpp::traits::input_parameter< double >::type thres(thresSEXP);
    Rcpp::traits::input_parameter< double >::type exp(expSEXP);
    rcpp_result_gen = Rcpp::wrap(fast_povgap(y, w, thres, exp));
    return rcpp_result_gen;
END_RCPP
}
// theil_l
double theil_l(NumericVector y, NumericVector w);
RcppExport SEXP _taxsimutilities_theil_l(SEXP ySEXP, SEXP wSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector >::type w(wSEXP);
    rcpp_result_gen = Rcpp::wrap(theil_l(y, w));
    return rcpp_result_gen;
END_RCPP
}
// theil_t
double theil_t(NumericVector y, NumericVector w);
RcppExport SEXP _taxsimutilities_theil_t(SEXP ySEXP, SEXP wSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector >::type w(wSEXP);
    rcpp_result_gen = Rcpp::wrap(theil_t(y, w));
    return rcpp_result_gen;
END_RCPP
}
// gen_ent
double gen_ent(NumericVector y, NumericVector w, double a);
RcppExport SEXP _taxsimutilities_gen_ent(SEXP ySEXP, SEXP wSEXP, SEXP aSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector >::type w(wSEXP);
    Rcpp::traits::input_parameter< double >::type a(aSEXP);
    rcpp_result_gen = Rcpp::wrap(gen_ent(y, w, a));
    return rcpp_result_gen;
END_RCPP
}
// atkinson_1
double atkinson_1(NumericVector y, NumericVector w);
RcppExport SEXP _taxsimutilities_atkinson_1(SEXP ySEXP, SEXP wSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector >::type w(wSEXP);
    rcpp_result_gen = Rcpp::wrap(atkinson_1(y, w));
    return rcpp_result_gen;
END_RCPP
}
// atkinson_e
double atkinson_e(NumericVector y, NumericVector w, double e);
RcppExport SEXP _taxsimutilities_atkinson_e(SEXP ySEXP, SEXP wSEXP, SEXP eSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericVector >::type w(wSEXP);
    Rcpp::traits::input_parameter< double >::type e(eSEXP);
    rcpp_result_gen = Rcpp::wrap(atkinson_e(y, w, e));
    return rcpp_result_gen;
END_RCPP
}
// greg_cpp
arma::mat greg_cpp(arma::colvec W, arma::mat& C, arma::vec& B, arma::colvec& L, arma::colvec& U);
RcppExport SEXP _taxsimutilities_greg_cpp(SEXP WSEXP, SEXP CSEXP, SEXP BSEXP, SEXP LSEXP, SEXP USEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< arma::colvec >::type W(WSEXP);
    Rcpp::traits::input_parameter< arma::mat& >::type C(CSEXP);
    Rcpp::traits::input_parameter< arma::vec& >::type B(BSEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type L(LSEXP);
    Rcpp::traits::input_parameter< arma::colvec& >::type U(USEXP);
    rcpp_result_gen = Rcpp::wrap(greg_cpp(W, C, B, L, U));
    return rcpp_result_gen;
END_RCPP
}
// run_sim_cpp_parallel
void run_sim_cpp_parallel(const int iters, const int M, const int N, const std::vector<int> in_matrix, const std::vector<double> U, const std::vector<double> V, const std::vector<int> ID, NumericMatrix out_matrix, const std::vector<double> cw, const std::vector<double> lambda, const int ncores, const int seed);
RcppExport SEXP _taxsimutilities_run_sim_cpp_parallel(SEXP itersSEXP, SEXP MSEXP, SEXP NSEXP, SEXP in_matrixSEXP, SEXP USEXP, SEXP VSEXP, SEXP IDSEXP, SEXP out_matrixSEXP, SEXP cwSEXP, SEXP lambdaSEXP, SEXP ncoresSEXP, SEXP seedSEXP) {
BEGIN_RCPP
    Rcpp::traits::input_parameter< const int >::type iters(itersSEXP);
    Rcpp::traits::input_parameter< const int >::type M(MSEXP);
    Rcpp::traits::input_parameter< const int >::type N(NSEXP);
    Rcpp::traits::input_parameter< const std::vector<int> >::type in_matrix(in_matrixSEXP);
    Rcpp::traits::input_parameter< const std::vector<double> >::type U(USEXP);
    Rcpp::traits::input_parameter< const std::vector<double> >::type V(VSEXP);
    Rcpp::traits::input_parameter< const std::vector<int> >::type ID(IDSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type out_matrix(out_matrixSEXP);
    Rcpp::traits::input_parameter< const std::vector<double> >::type cw(cwSEXP);
    Rcpp::traits::input_parameter< const std::vector<double> >::type lambda(lambdaSEXP);
    Rcpp::traits::input_parameter< const int >::type ncores(ncoresSEXP);
    Rcpp::traits::input_parameter< const int >::type seed(seedSEXP);
    run_sim_cpp_parallel(iters, M, N, in_matrix, U, V, ID, out_matrix, cw, lambda, ncores, seed);
    return R_NilValue;
END_RCPP
}
// test_int
NumericVector test_int(NumericVector x, NumericVector y);
RcppExport SEXP _taxsimutilities_test_int(SEXP xSEXP, SEXP ySEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< NumericVector >::type x(xSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    rcpp_result_gen = Rcpp::wrap(test_int(x, y));
    return rcpp_result_gen;
END_RCPP
}
// test_intm
void test_intm(const std::vector<float> x, const std::vector<float> y, NumericMatrix integral, const int M, const int N, const int ncores);
RcppExport SEXP _taxsimutilities_test_intm(SEXP xSEXP, SEXP ySEXP, SEXP integralSEXP, SEXP MSEXP, SEXP NSEXP, SEXP ncoresSEXP) {
BEGIN_RCPP
    Rcpp::traits::input_parameter< const std::vector<float> >::type x(xSEXP);
    Rcpp::traits::input_parameter< const std::vector<float> >::type y(ySEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type integral(integralSEXP);
    Rcpp::traits::input_parameter< const int >::type M(MSEXP);
    Rcpp::traits::input_parameter< const int >::type N(NSEXP);
    Rcpp::traits::input_parameter< const int >::type ncores(ncoresSEXP);
    test_intm(x, y, integral, M, N, ncores);
    return R_NilValue;
END_RCPP
}
// stable_point
Rcpp::NumericMatrix stable_point(const Rcpp::NumericMatrix U, const Rcpp::NumericMatrix w);
RcppExport SEXP _taxsimutilities_stable_point(SEXP USEXP, SEXP wSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type U(USEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type w(wSEXP);
    rcpp_result_gen = Rcpp::wrap(stable_point(U, w));
    return rcpp_result_gen;
END_RCPP
}
// stable_pointv
Rcpp::NumericVector stable_pointv(const Rcpp::NumericVector U, const Rcpp::NumericVector w);
RcppExport SEXP _taxsimutilities_stable_pointv(SEXP USEXP, SEXP wSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type U(USEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type w(wSEXP);
    rcpp_result_gen = Rcpp::wrap(stable_pointv(U, w));
    return rcpp_result_gen;
END_RCPP
}
// int_optv
double int_optv(const Rcpp::NumericVector re, const int row, const Rcpp::NumericMatrix hrs, const Rcpp::NumericMatrix disp, const Rcpp::NumericMatrix tw, const Rcpp::NumericVector mn, const Rcpp::NumericVector sdv);
RcppExport SEXP _taxsimutilities_int_optv(SEXP reSEXP, SEXP rowSEXP, SEXP hrsSEXP, SEXP dispSEXP, SEXP twSEXP, SEXP mnSEXP, SEXP sdvSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type re(reSEXP);
    Rcpp::traits::input_parameter< const int >::type row(rowSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type hrs(hrsSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type disp(dispSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type tw(twSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type mn(mnSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type sdv(sdvSEXP);
    rcpp_result_gen = Rcpp::wrap(int_optv(re, row, hrs, disp, tw, mn, sdv));
    return rcpp_result_gen;
END_RCPP
}
// int_optrc
double int_optrc(const Rcpp::NumericVector re, const Rcpp::NumericVector hrs1, const Rcpp::NumericVector hrs2, const Rcpp::NumericVector disp, const Rcpp::NumericVector tw, const Rcpp::NumericVector mn);
RcppExport SEXP _taxsimutilities_int_optrc(SEXP reSEXP, SEXP hrs1SEXP, SEXP hrs2SEXP, SEXP dispSEXP, SEXP twSEXP, SEXP mnSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type re(reSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type hrs1(hrs1SEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type hrs2(hrs2SEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type disp(dispSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type tw(twSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type mn(mnSEXP);
    rcpp_result_gen = Rcpp::wrap(int_optrc(re, hrs1, hrs2, disp, tw, mn));
    return rcpp_result_gen;
END_RCPP
}
// int_optr
double int_optr(const Rcpp::NumericVector re, const Rcpp::NumericVector hrs, const Rcpp::NumericVector disp, const Rcpp::NumericVector tw, const Rcpp::NumericVector mn);
RcppExport SEXP _taxsimutilities_int_optr(SEXP reSEXP, SEXP hrsSEXP, SEXP dispSEXP, SEXP twSEXP, SEXP mnSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type re(reSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type hrs(hrsSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type disp(dispSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type tw(twSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type mn(mnSEXP);
    rcpp_result_gen = Rcpp::wrap(int_optr(re, hrs, disp, tw, mn));
    return rcpp_result_gen;
END_RCPP
}
// recalc_quants
void recalc_quants(NumericMatrix optp, const Rcpp::NumericMatrix hrs, const Rcpp::NumericMatrix disp, const Rcpp::NumericMatrix tw, const Rcpp::NumericVector mn, const int N, const int M);
RcppExport SEXP _taxsimutilities_recalc_quants(SEXP optpSEXP, SEXP hrsSEXP, SEXP dispSEXP, SEXP twSEXP, SEXP mnSEXP, SEXP NSEXP, SEXP MSEXP) {
BEGIN_RCPP
    Rcpp::traits::input_parameter< NumericMatrix >::type optp(optpSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type hrs(hrsSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type disp(dispSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type tw(twSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type mn(mnSEXP);
    Rcpp::traits::input_parameter< const int >::type N(NSEXP);
    Rcpp::traits::input_parameter< const int >::type M(MSEXP);
    recalc_quants(optp, hrs, disp, tw, mn, N, M);
    return R_NilValue;
END_RCPP
}
// int_optvc
double int_optvc(const Rcpp::NumericVector re, const int row, const Rcpp::NumericVector h1, const Rcpp::NumericMatrix hrs1, const Rcpp::NumericVector h2, const Rcpp::NumericMatrix hrs2, const Rcpp::NumericVector y1, const Rcpp::NumericMatrix disp, const Rcpp::NumericMatrix tw, const Rcpp::NumericVector mn, const Rcpp::NumericVector std);
RcppExport SEXP _taxsimutilities_int_optvc(SEXP reSEXP, SEXP rowSEXP, SEXP h1SEXP, SEXP hrs1SEXP, SEXP h2SEXP, SEXP hrs2SEXP, SEXP y1SEXP, SEXP dispSEXP, SEXP twSEXP, SEXP mnSEXP, SEXP stdSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type re(reSEXP);
    Rcpp::traits::input_parameter< const int >::type row(rowSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type h1(h1SEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type hrs1(hrs1SEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type h2(h2SEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type hrs2(hrs2SEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type y1(y1SEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type disp(dispSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericMatrix >::type tw(twSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type mn(mnSEXP);
    Rcpp::traits::input_parameter< const Rcpp::NumericVector >::type std(stdSEXP);
    rcpp_result_gen = Rcpp::wrap(int_optvc(re, row, h1, hrs1, h2, hrs2, y1, disp, tw, mn, std));
    return rcpp_result_gen;
END_RCPP
}

static const R_CallMethodDef CallEntries[] = {
    {"_taxsimutilities_greg_cpp_one", (DL_FUNC) &_taxsimutilities_greg_cpp_one, 3},
    {"_taxsimutilities_fast_bs_sum", (DL_FUNC) &_taxsimutilities_fast_bs_sum, 1},
    {"_taxsimutilities_logSumExp", (DL_FUNC) &_taxsimutilities_logSumExp, 2},
    {"_taxsimutilities_stable_ig", (DL_FUNC) &_taxsimutilities_stable_ig, 4},
    {"_taxsimutilities_stable_n", (DL_FUNC) &_taxsimutilities_stable_n, 4},
    {"_taxsimutilities_stable_p", (DL_FUNC) &_taxsimutilities_stable_p, 4},
    {"_taxsimutilities_llc_cpp", (DL_FUNC) &_taxsimutilities_llc_cpp, 14},
    {"_taxsimutilities_llc_alt_cpp", (DL_FUNC) &_taxsimutilities_llc_alt_cpp, 14},
    {"_taxsimutilities_lls_cpp", (DL_FUNC) &_taxsimutilities_lls_cpp, 10},
    {"_taxsimutilities_lls_alt_cpp", (DL_FUNC) &_taxsimutilities_lls_alt_cpp, 10},
    {"_taxsimutilities_llsopt_cpp", (DL_FUNC) &_taxsimutilities_llsopt_cpp, 11},
    {"_taxsimutilities_fast_group_sum", (DL_FUNC) &_taxsimutilities_fast_group_sum, 2},
    {"_taxsimutilities_fast_group_max", (DL_FUNC) &_taxsimutilities_fast_group_max, 2},
    {"_taxsimutilities_fast_group_min", (DL_FUNC) &_taxsimutilities_fast_group_min, 2},
    {"_taxsimutilities_gini_sorted", (DL_FUNC) &_taxsimutilities_gini_sorted, 2},
    {"_taxsimutilities_fast_poverty", (DL_FUNC) &_taxsimutilities_fast_poverty, 4},
    {"_taxsimutilities_fast_med", (DL_FUNC) &_taxsimutilities_fast_med, 2},
    {"_taxsimutilities_fast_pov", (DL_FUNC) &_taxsimutilities_fast_pov, 3},
    {"_taxsimutilities_fast_cpov", (DL_FUNC) &_taxsimutilities_fast_cpov, 4},
    {"_taxsimutilities_fast_povgap", (DL_FUNC) &_taxsimutilities_fast_povgap, 4},
    {"_taxsimutilities_theil_l", (DL_FUNC) &_taxsimutilities_theil_l, 2},
    {"_taxsimutilities_theil_t", (DL_FUNC) &_taxsimutilities_theil_t, 2},
    {"_taxsimutilities_gen_ent", (DL_FUNC) &_taxsimutilities_gen_ent, 3},
    {"_taxsimutilities_atkinson_1", (DL_FUNC) &_taxsimutilities_atkinson_1, 2},
    {"_taxsimutilities_atkinson_e", (DL_FUNC) &_taxsimutilities_atkinson_e, 3},
    {"_taxsimutilities_greg_cpp", (DL_FUNC) &_taxsimutilities_greg_cpp, 5},
    {"_taxsimutilities_run_sim_cpp_parallel", (DL_FUNC) &_taxsimutilities_run_sim_cpp_parallel, 12},
    {"_taxsimutilities_test_int", (DL_FUNC) &_taxsimutilities_test_int, 2},
    {"_taxsimutilities_test_intm", (DL_FUNC) &_taxsimutilities_test_intm, 6},
    {"_taxsimutilities_stable_point", (DL_FUNC) &_taxsimutilities_stable_point, 2},
    {"_taxsimutilities_stable_pointv", (DL_FUNC) &_taxsimutilities_stable_pointv, 2},
    {"_taxsimutilities_int_optv", (DL_FUNC) &_taxsimutilities_int_optv, 7},
    {"_taxsimutilities_int_optrc", (DL_FUNC) &_taxsimutilities_int_optrc, 6},
    {"_taxsimutilities_int_optr", (DL_FUNC) &_taxsimutilities_int_optr, 5},
    {"_taxsimutilities_recalc_quants", (DL_FUNC) &_taxsimutilities_recalc_quants, 7},
    {"_taxsimutilities_int_optvc", (DL_FUNC) &_taxsimutilities_int_optvc, 11},
    {NULL, NULL, 0}
};

RcppExport void R_init_taxsimutilities(DllInfo *dll) {
    R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
    R_useDynamicSymbols(dll, FALSE);
}
