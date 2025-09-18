USE [telecom]
GO

TRUNCATE TABLE voda


-- DATA LOAD 
-- 1. Drop existing tables if they exist
DROP TABLE IF EXISTS voda_staging;

-- 2. Create staging table for raw CSV data
CREATE TABLE voda_staging  (
    Year INT,
    Quarter VARCHAR(5),
    Segment VARCHAR(100),
    Region VARCHAR(100),
    Subscribers BIGINT,
    Revenue_QR_Mn DECIMAL(18,2),
    ARPU_QR DECIMAL(18,2),
    Data_Usage_GB DECIMAL(18,2),
    Voice_Minutes_Mn DECIMAL(18,2),
    Customer_Satisfaction_Index DECIMAL(5,2),
    Churn_Rate_Pct DECIMAL(5,2),
    Complaints_Resolved_Pct DECIMAL(5,2),
    New_Activations BIGINT,
    Deactivations BIGINT,
    Avg_Resolution_Time_Hours DECIMAL(5,2),
    eSIM_Activations BIGINT,
    Recharge_Transactions_Thousand BIGINT,
    Retail_Store_Visits_Thousand BIGINT,
    Digital_Sales_Pct DECIMAL(5,2),
    Network_Availability_Pct DECIMAL(5,2)
);




-- 3. Bulk insert into staging table
BULK INSERT voda_staging
FROM 'D:\Internship\Project 4a (Vodafone Qatar)\Vodafone qatar dataset.csv'
WITH (
    FIRSTROW = 2,
    FIELDTERMINATOR = ',',
    ROWTERMINATOR = '\n',
    CODEPAGE = '65001',
    TABLOCK
);


-- 4. Create final clean table
DROP TABLE IF EXISTS voda


CREATE TABLE voda  (
    Year INT,
    Quarter VARCHAR(5),
    Segment VARCHAR(100),
    Region VARCHAR(100),
    Subscribers BIGINT,
    Revenue_QR_Mn DECIMAL(18,2),
    ARPU_QR DECIMAL(18,2),
    Data_Usage_GB DECIMAL(18,2),
    Voice_Minutes_Mn DECIMAL(18,2),
    Customer_Satisfaction_Index DECIMAL(5,2),
    Churn_Rate_Pct DECIMAL(5,2),
    Complaints_Resolved_Pct DECIMAL(5,2),
    New_Activations BIGINT,
    Deactivations BIGINT,
    Avg_Resolution_Time_Hours DECIMAL(5,2),
    eSIM_Activations BIGINT,
    Recharge_Transactions_Thousand BIGINT,
    Retail_Store_Visits_Thousand BIGINT,
    Digital_Sales_Pct DECIMAL(5,2),
    Network_Availability_Pct DECIMAL(5,2)
);


-- 5. Insert clean records from staging to final table


Insert into voda (
    Year ,
    Quarter ,
    Segment ,
    Region ,
    Subscribers,
    Revenue_QR_Mn ,
    ARPU_QR ,
    Data_Usage_GB,
    Voice_Minutes_Mn ,
    Customer_Satisfaction_Index ,
    Churn_Rate_Pct,
    Complaints_Resolved_Pct,
    New_Activations,
    Deactivations,
    Avg_Resolution_Time_Hours ,
    eSIM_Activations,
    Recharge_Transactions_Thousand,
    Retail_Store_Visits_Thousand,
    Digital_Sales_Pct,
    Network_Availability_Pct
	)
	SELECT
    -- Time fields
    Year, Quarter,
    -- Text cleanup
    LTRIM(RTRIM(Segment)) AS Segment,
    LTRIM(RTRIM(Region)) AS Region,

    -- Volume / financial metrics (no negatives allowed)
    CASE WHEN Subscribers < 0 THEN 0 ELSE Subscribers END,
    CASE WHEN Revenue_QR_Mn < 0 THEN 0 ELSE Revenue_QR_Mn END,
    CASE WHEN ARPU_QR < 0 THEN 0 ELSE ARPU_QR END,
    CASE WHEN Data_Usage_GB < 0 THEN 0 ELSE Data_Usage_GB END,
    CASE WHEN Voice_Minutes_Mn < 0 THEN 0 ELSE Voice_Minutes_Mn END,

    -- Percentages capped between 0–100
    CASE WHEN Customer_Satisfaction_Index < 0 THEN 0
         WHEN Customer_Satisfaction_Index > 100 THEN 100
         ELSE Customer_Satisfaction_Index END,

    CASE WHEN Churn_Rate_Pct < 0 THEN 0
         WHEN Churn_Rate_Pct > 100 THEN 100
         ELSE Churn_Rate_Pct END,

    CASE WHEN Complaints_Resolved_Pct < 0 THEN 0
         WHEN Complaints_Resolved_Pct > 100 THEN 100
         ELSE Complaints_Resolved_Pct END,

    -- Operational KPIs
    CASE WHEN New_Activations < 0 THEN 0 ELSE New_Activations END,
    CASE WHEN Deactivations < 0 THEN 0 ELSE Deactivations END,
    CASE WHEN Avg_Resolution_Time_Hours < 0 THEN NULL ELSE Avg_Resolution_Time_Hours END,
    CASE WHEN eSIM_Activations < 0 THEN 0 ELSE eSIM_Activations END,
    CASE WHEN Recharge_Transactions_Thousand < 0 THEN 0 ELSE Recharge_Transactions_Thousand END,
    CASE WHEN Retail_Store_Visits_Thousand < 0 THEN 0 ELSE Retail_Store_Visits_Thousand END,

    -- Digital & network metrics
    CASE WHEN Digital_Sales_Pct < 0 THEN 0
         WHEN Digital_Sales_Pct > 100 THEN 100
         ELSE Digital_Sales_Pct END,

    CASE WHEN Network_Availability_Pct < 0 THEN 0
         WHEN Network_Availability_Pct > 100 THEN 100
         ELSE Network_Availability_Pct END

FROM (
    -- Deduplicate on Year + Quarter + Segment + Region
    SELECT *,
           ROW_NUMBER() OVER (PARTITION BY Year, Quarter, Segment, Region ORDER BY (SELECT NULL)) AS rn
    FROM voda_staging
) s
WHERE rn = 1
  AND Year IS NOT NULL
  AND Quarter IS NOT NULL;

