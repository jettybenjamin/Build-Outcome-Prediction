import pandas as pd
import matplotlib.pyplot as plt

# Creating a DataFrame from the provided data
data = {
    'Project': ['cloudify', 'graylog', 'jackrabbit-oak', 'metasploit-framework', 'open-build-service', 'openproject', 'rails', 'ruby', 'sonarqube'],
    'LSTM-GRU': [72, 89, 71, 92, 72, 63, 64, 79, 75],
    'DT': [52, 65, 49, 65, 52, 57, 42, 42, 34],
    'LR': [74, 42, 52, 72, 65, 63, 59, 46, 52],
    'ADA': [39, 66, 53, 76, 47, 60, 46, 60, 58],
    'RF': [66, 54, 53, 79, 43, 62, 53, 63, 46],
    'SVC': [29, 23, 53, 41, 63, 58, 35, 31, 28]
}

df = pd.DataFrame(data)

# Creating a boxplot
plt.boxplot(df.drop('Project', axis=1).values, vert=True, patch_artist=True)

# Adding labels and title
plt.xticks(range(1, len(df.columns)), df.columns[1:])
plt.xlabel('Methods')
plt.ylabel('Accuracy')
plt.title('Boxplot of Build Outcome Prediction Accuracy for Various Projects')

# Show the plot
plt.show()
