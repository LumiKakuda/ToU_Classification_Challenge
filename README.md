<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sepsis Survival Minimal Clinical Records</title>
    <style>
        body { font-family: Arial, sans-serif; line-height: 1.6; max-width: 900px; margin: 0 auto; padding: 20px; }
        h1 { border-bottom: 2px solid #333; padding-bottom: 10px; }
        h2 { color: #0056b3; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 30px; }
        h3 { color: #333; margin-top: 20px; }
        table { border-collapse: collapse; width: 100%; margin: 15px 0; }
        th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
        th { background-color: #f2f2f2; }
        ul { list-style-type: disc; margin-left: 20px; }
        .emphasis { font-weight: bold; }
    </style>
</head>
<body>

    <h1>Classification of Sepsis Survival Minimal Clinical Records</h1>

    <h2>1. Project Outline</h2>
    <p>This project focuses on developing and optimizing a **regression model** to predict the quantitative **Stroke Risk Percentage** for individuals based on their health profiles and reported symptoms. This model aims to serve as a proactive tool for medical professionals to assess risk, enabling targeted preventative care.</p>

    <h3>Problem Definition</h3>
    <p>The core challenge is a **regression task**: accurately estimating a continuous numerical value (stroke risk percentage) to provide a precise, quantitative risk assessment.</p>

    <h3>Expected Social Impact (SDG 3)</h3>
    <p>The project aligns with **SDG 3: Good Health and Well-being** by supporting:</p>
    <ul>
        <li>**Preventative Healthcare:** Facilitating early identification and intervention for high-risk individuals.</li>
        <li>**Resource Optimization:** Helping healthcare systems allocate resources efficiently by focusing on populations identified as high-risk.</li>
    </ul>

    <h2>2. Dataset and Preprocessing</h2>

    <h3>Dataset Details</h3>
    <p>The analysis utilizes the **Stroke Risk Prediction Dataset Based on Symptoms**, a <span class="emphasis">synthetic dataset</span> available on Kaggle (<a href="https://www.kaggle.com/datasets/mahatiratusher/stroke-risk-prediction-dataset">Kaggle Link</a>). The data's structure reflects established medical risk factors for controlled model development.</p>
    <ul>
        <li><strong>Target:</strong> <code>Stroke Risk (%)</code> (continuous value).</li>
        <li><strong>Feature Types:</strong> 1 numerical feature (<code>Age</code>) and 15 binary (0/1) symptom/condition features.</li>
    </ul>

    <h3>Data Preparation Notes</h3>
    <ul>
        <li>The column <code>"At Risk (Binary)"</code> was dropped as it represents a classification target.</li>
        <li>A total of **2,033 duplicate rows** were removed to ensure the model's training integrity.</li>
    </ul>

    <h2>3. Instructions to Run the Code</h2>

    <h3>Required Project Files</h3>
    <p>To successfully run the project's Jupyter Notebook, you must ensure the following files are saved in the **same directory** as your notebook file:</p>
    <ul>
        <li>The **dataset** file downloaded from the provided Kaggle link.</li>
        <li>The external Python functions file: <strong><code>Functions_EAD.py</code></strong>.</li>
    </ul>

    <h3>Required Libraries</h3>
    <p>The project requires the following Python libraries. They can be installed via a package manager like <code>pip</code>:</p>
    <table>
        <thead>
            <tr>
                <th>Library</th>
                <th>Description</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><code>pandas</code>, <code>numpy</code></td>
                <td>Data manipulation and numerical operations.</td>
            </tr>
            <tr>
                <td><code>scikit-learn</code></td>
                <td>Machine learning models (Random Forest, Decision Tree, Linear Regression), cross-validation, and metrics.</td>
            </tr>
            <tr>
                <td><code>seaborn</code>, <code>matplotlib.pyplot</code></td>
                <td>Data visualization and plotting.</td>
            </tr>
        </tbody>
    </table>

    <h3>Execution</h3>
    <p>Once all required files and libraries are in place, open the Jupyter Notebook. All preprocessing steps, model training (including hyperparameter tuning), and cross-validation evaluations are contained within the notebook and can be executed sequentially.</p>

    <h2>4. Project Maintainer</h2>
    <p><strong>[Your Name/Alias]</strong><br>
    [Your Contact Information or GitHub Profile Link]</p>

</body>
</html>
