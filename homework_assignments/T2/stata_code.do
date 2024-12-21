clear all
cd "data"


import excel "database_final.xlsx", firstrow

gen month = mofd(date)
tsset month, monthly


varsoc copper_returns spot_returns, maxlag(12)

var copper_returns spot_returns, lags(1)
gen spot_returns_f1 = F.spot_returns  // Forward one period
gen spot_returns_f2 = F2.spot_returns // Forward two periods

vargranger


//
// Non-Granger Causality Tests
//


gen copper_returns_f1 = F.copper_returns    // Forward one period
gen spot_returns_f1 = F.spot_returns    // Forward one period


// Sims causality

newey copper_returns l.spot_returns spot_returns spot_returns_f1, lag(0)
newey spot_returns l.copper_returns copper_returns copper_returns_f1, lag(0)


// Geweke, Meese & Dent causality

newey copper_returns l.copper_returns l.spot_returns spot_returns spot_returns_f1, lag(0)
newey spot_returns l.spot_returns l.copper_returns copper_returns copper_returns_f1, lag(0)


