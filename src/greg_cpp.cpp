#include <RcppArmadillo.h>

using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]

// This is a simple example of exporting a C++ function to R. You can
// source this function into an R session using the Rcpp::sourceCpp 
// function (or via the Source button on the editor toolbar). Learn
// more about Rcpp at:
//
//   http://www.rcpp.org/
//   http://adv-r.had.co.nz/Rcpp.html
//   http://gallery.rcpp.org/
//

//' @export
// [[Rcpp::export]]
arma::mat greg_cpp(arma::colvec W,
                       arma::mat& C,
                       arma::vec& B,
                       arma::colvec& L,
                       arma::colvec& U) {
  float diff = sum(B);
  arma::mat ew(size(C));
  arma::vec iw = B - C*W;
  float new_diff = sum((iw % iw)/B);
  int it = 0;
  arma::rowvec Wt(ew.n_cols);
  
  while (new_diff < diff - 0.01 && it < 100){
    it++;
    diff = new_diff;
    Wt = W.t();
    ew.each_row() = Wt;
    W = max(L, min(U, W % (1 + (solve((C % ew) * C.t(), iw).t() * C)).t()));
    iw = B - C*W;
    new_diff = sum((iw % iw)/B);
  }
  return(W);
}
