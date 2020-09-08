# FIFA 19 Player Position Predictor
Computational Intelligence Fundamentals project, Software Engineering and Information Technologies, FTN, 2019

SW-39/2016, Uroš Ogrizović

Technologies used: Keras, Python

# Overview

The goal is to predict the position of a player in the video game FIFA 19 based on the player's attributes. As the [dataset](https://www.kaggle.com/karangadiya/fifa19) is unbalanced, predicting a player's position with high accuracy was more difficult than expected. Hence, positions were grouped into sections, which made the dataset more balanced, and the predictions more accurate.

# Data

## Players per position

<table align="center">
  <thead>
    <th>position</th>
    <th>LS</th>
    <th>ST</th>
    <th>RS</th>
    <th>LW</th>
    <th>LF</th>
    <th>CF</th>
    <th>RF</th>
    <th>RW</th>
    <th>LAM</th>
    <th>CAM</th>
    <th>RAM</th>
    <th>LM</th>
    <th>LCM</th>
    <th>CM</th>
    <th>RCM</th>
    <th>RM</th>
    <th>LWB</th>
    <th>LDM</th>
    <th>CDM</th>
    <th>RDM</th>
    <th>RWB</th>
    <th>LB</th>
    <th>LCB</th>
    <th>CB</th>
    <th>RCB</th>
    <th>RB</th>
    <th>GK</th>
  </thead>
  <tbody align="center">
    <tr>
      <td>#</td>
      <td style="text-align:center">170</td>
      <td style="text-align:center">1726</td>
      <td style="text-align:center">162</td>
      <td style="text-align:center">303</td>
      <td style="text-align:center">13</td>
      <td style="text-align:center">59</td>
      <td style="text-align:center">12</td>
      <td style="text-align:center">293</td>
      <td style="text-align:center">16</td>
      <td style="text-align:center">787</td>
      <td style="text-align:center">17</td>
      <td style="text-align:center">864</td>
      <td style="text-align:center">302</td>
      <td style="text-align:center">1112</td>
      <td style="text-align:center">305</td>
      <td style="text-align:center">885</td>
      <td style="text-align:center">62</td>
      <td style="text-align:center">194</td>
      <td style="text-align:center">764</td>
      <td style="text-align:center">206</td>
      <td style="text-align:center">74</td>
      <td style="text-align:center">1050</td>
      <td style="text-align:center">520</td>
      <td style="text-align:center">1446</td>
      <td style="text-align:center">528</td>
      <td style="text-align:center">1029</td>
      <td style="text-align:center">1618</td>
    </tr>
    <tr>
      <td>%</td>
      <td style="text-align:center">1.17</td>
      <td style="text-align:center">11.89</td>
      <td style="text-align:center">1.16</td>
      <td style="text-align:center">2.09</td>
      <td style="text-align:center">0.09</td>
      <td style="text-align:center">0.41</td>
      <td style="text-align:center">0.08</td>
      <td style="text-align:center">2.02</td>
      <td style="text-align:center">0.11</td>
      <td style="text-align:center">5.42</td>
      <td style="text-align:center">0.12</td>
      <td style="text-align:center">5.95</td>
      <td style="text-align:center">2.08</td>
      <td style="text-align:center">7.66</td>
      <td style="text-align:center">2.10</td>
      <td style="text-align:center">6.10</td>
      <td style="text-align:center">0.43</td>
      <td style="text-align:center">1.34</td>
      <td style="text-align:center">5.26</td>
      <td style="text-align:center">1.42</td>
      <td style="text-align:center">0.50</td>
      <td style="text-align:center">7.23</td>
      <td style="text-align:center">3.58</td>
      <td style="text-align:center">9.96</td>
      <td style="text-align:center">3.64</td>
      <td style="text-align:center">7.09</td>
      <td style="text-align:center">11.15</td>
    </tr>
  </tbody>
</table>

## Players per section

# Models
## ANN
## Random Forest.
