# CV bullets

- Built a tested Python option-pricing library (Black–Scholes–Merton Greeks, CRR American trees, Monte Carlo) that reproduces Hull's textbook values; combining antithetic and control variates cut Monte Carlo variance 23×.
- Constructed a SPY implied-volatility surface from 9,280 live option quotes (2,688 after cleaning) with an SVI fit accurate to 0.13 vol points on held-out strikes, showing a 13.7-vol-point one-month put skew.
- Back-tested delta-hedging a short one-month SPY call over 5,465 windows since 2005, explaining 91% of P&L variation with gamma–theta theory and showing that hedging error shrinks with √Δt (slope 0.49) only when the hedge uses realised volatility.
