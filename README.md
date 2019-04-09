# Applying Machine Learning in Stock Price Trend Forecasting

## 1. Introduction
Stock prices fluctuate within seconds and are affected by complicated financial and non-financial indicators. Hence, stock prices prediction is an ambitious project. However, Thanks to Machine Learning techniques, we have the capacity of identifying the stock trend from massive amounts of data that capture the underlying stock price dynamics. <br/>

In this project, we utilized past data, technical indicators, and economic indexes and applied supervised learning methods to stock price trend forecasting on the next trading week. <br/>

As opposed to predicting the trend in short-term which is used in the high-frequency trading market, we intend to forecast the upward and downward movement in the weekly-basis not solely for algorithmic trading, but as a supplement to help investors alike on decision-making.<br/>

## 2. Project Plan
##### Data Acquisition
  The training data used in our project were collected from Alpha Vantage (https://www.alphavantage.co), a Silicon Valley-based “leading provider of free APIs for real-time and historical data on stocks, forex (FX), and digital/cryptocurrencies”.<br/>
  
  Their free APIs allow us to retrieve the history of stock price changes in the past 20+ years which is updated every five minutes. While we can increase the timeliness of the data by upgrading to their paid data which is updated every few seconds, as our intention is to predict the market in the long-term, we settle with the free plan as the interval of five minutes is sufficient.<br/>
  
  Additionally, Alpha Vantage APIs also allow us to retrieve a wide number of technical indicators, such as simple moving average (SMA), weighted average price (VWAP), etc. We expect to use these technical indicators to improve the accuracy of our prediction.<br/>
  
  Lastly, as we consider that the changes in the Consumer Price Index (CPI) and the Prime Interest Rate might affect the stock price trend, we plan to incorporate these numbers in our measure. The data for these indexes can be retrieved from the Wall Street Journal Money Rate (http://www.wsj.com/mdc/public/page/2_3020-moneyrate.html) and the Bureau of Labor Statistics (https://www.bls.gov/cpi/data.htm) 

##### Attributes Determination
  As mentioned above, we will use the data from Alpha Vintage. For the historical data, it will consist of five main attributes: the opening price, the highest price, the low price, the closing price and the volume of exchanges. At the time of this proposal, we consider the closing price to be the most significant and intend to use as the main indicator of our models.<br/>
  
  Our factors like the provided technical indicators or the Consumer Price Index will be used but at different levels of importance. We will work together to decide how relevant are these factors as we progress through the project.<br/>
  
  Non-quantitative factors such as economic shocks, political changes, natural and artificial disasters, and market psychology might also affect the market. But as we only intend the Machine Learning models to be used to assist humans to invest, not to invest by itself, we believe the investors are responsible to add these factors into their decision-making process. Hence, we leave the factors outside the scope of this project.<br/>
  
  Nevertheless, we hope to use as many as features in predicting the stock price trend as we believe it will provide better accuracy.

##### Machine Learning Approach
  At the basic level of our project, we plan to use the Logistic Regression model for evaluating data and predicting the trend. But if our time and capacity are sufficient, we will try to explore more advanced methods to give better results such as Support Vector Machine (SVC), K-Nearest Neighbors or Random Forests.

## 3. Outcome
The intended outcome of this project is to produce a practical trading model that can accurately predict the stock price trend in the long-term (weekly basis). <br/>

Additionally, if time permits, we hope to build a user interface on top of the produced model which non-technical users can easily access and utilize through their web browsers.

## 4. References
- [Stock Price Forecasting using Machine Learning Supervised Methods, Sharvil Katariya and Saurabh Jain](https://github.com/scorpionhiccup/StockPricePrediction/blob/master/Report.pdf)
- [How can machine learning help stock investment?, Xin Guo](http://cs229.stanford.edu/proj2015/009_report.pdf)
