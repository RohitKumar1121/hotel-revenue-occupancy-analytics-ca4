# Hotel Revenue & Occupancy Analytics Dashboard (CA4 Capstone Project #37)

**Course:** IBA – 5th Semester | Business Analytics (CA4 Capstone Project)  
**Project Number & Title:** Project No. 37 – Hotel Revenue & Occupancy Analytics  
**Domain:** Hospitality Analytics  
**Technology Stack:** Python, Pandas, NumPy, Matplotlib, Seaborn, Plotly, Streamlit, GitHub  

---

## 1. Business Problem & Project Objective

A hotel chain operating both an urban **City Hotel** (Lisbon) and a leisure **Resort Hotel** (Algarve) wants to understand occupancy patterns, room revenue generation, booking channels, customer segments, and the drivers of booking cancellations. High cancellation rates and heavy reliance on third-party Online Travel Agencies (OTAs) erode net realized revenue and complicate capacity planning.

**Core Objectives:**
1. Quantify realized room revenue versus lost revenue from cancellations across both properties (`City Hotel` vs. `Resort Hotel`).
2. Evaluate booking channel profitability and cancellation exposure across `Online TA`, `Offline TA/TO`, `Direct`, `Groups`, and `Corporate` segments.
3. Identify behavioral predictors of booking cancellations—including booking lead time, special requests, deposit structures, and room reassignment.
4. Analyze seasonal demand curves (summer peak vs. winter trough) and top international vs. domestic source markets.
5. Deliver an interactive **Streamlit Decision-Support Dashboard** with executive KPIs, multi-dimensional filters, and evidence-backed revenue management recommendations.

---

## 2. Real Public-Domain Dataset Source & Verification

In strict compliance with CA4 Capstone Project guidelines prohibiting synthetic or AI-generated data, this project analyzes a **100% real-world, peer-reviewed hospitality dataset** extracted from actual Property Management Systems (PMS) in Portugal:

- **Dataset Name:** Hotel Booking Demand Dataset (`hotel_bookings.csv`)
- **Time Horizon:** July 1, 2015 to August 31, 2017 (`119,390` raw PMS booking records across `32` attributes; `87,228` unique cleaned records after deduplication and anomaly removal)
- **Primary Academic Citation:** Antonio, N., de Almeida, A., & Nunes, L. (2019). *Hotel booking demand datasets.* **Data in Brief**, Volume 22, pp. 41–49.
- **Official Verification Links:**
  - [Peer-Reviewed Journal Article (NIH PMC Full Text)](https://pmc.ncbi.nlm.nih.gov/articles/PMC6297060/)
  - [Public Dataset Repository (Kaggle Mirror)](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand)

---

## 3. Repository Structure

```text
├── app.py                                         # Interactive Streamlit Dashboard Application (Step 5)
├── CA4_Hotel_Revenue_Occupancy_Analysis.ipynb     # End-to-End Python EDA & Data Cleaning Notebook (Steps 1–4)
├── hotel_bookings.csv.zip                         # Real Public-Domain Dataset (119,390 rows × 32 columns)
├── requirements.txt                               # Python Package Dependencies for Streamlit Cloud Deployment
└── README.md                                      # Project Overview, Methodology & Setup Instructions
```

---

## 4. Key Analytical Findings

1. **Massive Revenue Leakage from Cancellations:** Across the `87,228` unique valid bookings (`€34.45M` gross booking value), **27.52%** (`24,008` bookings) were canceled, resulting in **`€22.97M` in realized room revenue** and **`€11.48M` in lost room revenue** (33.3% revenue leakage). In the full un-deduplicated PMS log (`119,208` valid records), the cancellation rate is **37.08%** (`€16.72M` lost).
2. **Channel Asymmetry (Online TA vs. Direct):** `Online TA` drives **59.1%** of unique booking volume (`51,555` bookings) at a high Average Daily Rate (**€118.29**), but suffers a **35.39% cancellation rate**, accounting for **80.9% (`€9.29M`)** of all canceled room revenue. Meanwhile, the **Direct** channel achieves a nearly identical ADR (**€116.78**) with less than half the cancellation rate (**14.75%**) and zero OTA commission fees.
3. **Lead-Time Risk Escalation:** Cancellation probability increases nearly fivefold with booking horizon—rising from **8.42%** for last-minute bookings (`0–7 days`) to **25.39%** (`8–30 days`), **32.04%** (`31–90 days`), **35.01%** (`91–180 days`), and **39.74%** for bookings made more than `180 days` in advance.
4. **Special Requests as a Commitment Indicator:** Guests submitting **0 special requests** cancel at **33.25%**, whereas guests submitting **1 request** cancel at **22.44%** and those with **3+ requests** cancel at **16.25%**—demonstrating that pre-arrival engagement strongly signals stay commitment.
5. **Domestic vs. International Reliability:** Domestic Portuguese bookings (`PRT`, `27,354` unique bookings) cancel at **35.76%**, whereas inbound international air travelers from the **United Kingdom (`GBR`, 19.03%)**, **France (`FRA`, 19.63%)**, and **Germany (`DEU`, 19.55%)** exhibit substantially lower cancellation rates.

---

## 5. How to Run Locally & Deploy on Streamlit Community Cloud

### Local Execution
1. Clone this repository and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Launch the interactive Streamlit dashboard:
   ```bash
   streamlit run app.py
   ```
3. Open `http://localhost:8501` in your browser. Ensure `hotel_bookings.csv` or `hotel_bookings.csv.zip` is in the project root directory.

### Deployment on Streamlit Community Cloud (Step 9)
1. Push all files (`app.py`, `requirements.txt`, `README.md`, and `hotel_bookings.csv.zip`) to a public GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) using your GitHub account.
3. Click **Create app**, select your repository, branch (`main`), and main file path (`app.py`), and click **Deploy**.
