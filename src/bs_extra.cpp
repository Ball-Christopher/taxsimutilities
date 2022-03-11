#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

//' @export
// [[Rcpp::export]]
arma::mat greg_cpp_one(arma::colvec W,
                       arma::mat& C,
                       arma::vec& B) {

  arma::mat ew(size(C));
  ew.each_row() = W.t();
  
  W = W % (1 + (solve((C % ew) * C.t(), B - C*W).t() * C)).t();

  return(W);
}

//' @export
// [[Rcpp::export]]
NumericVector fast_bs_sum(NumericVector pattern) {
  
  NumericVector xx( pattern.length() );
  
  for (NumericVector::iterator i = pattern.begin(); 
       i != pattern.end(); ++i){
    xx[*i - 1]++;
  }
  
  return( xx );
}