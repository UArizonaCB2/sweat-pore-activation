import pandas as pd
import matplotlib.pyplot as plt
import re, os 

def get_EDA_data(file):
    # Read the excel file 
    df = pd.read_excel(f'VideoProcessing/EDA_data/{file}')
    
    # Create scatter plot - EDA
    # plt.figure(figsize=(10, 6))
    # plt.scatter(df['Column1'], df['Column2'])
    # plt.xlabel('X Label')
    # plt.ylabel('Y Label')
    # plt.title('EDA data')
    # plt.show()
    time_points = df['Column1'].values
    eda_values = df['Column2'].values
    
    return time_points, eda_values

def get_prediction_data():
    """
    Parse prediction data from prediction.txt file.
    Returns a list of tuples containing (frame_number, pore_count).
    """
    # text = "frame0171 have Pores [label1]: 175"
    data = [] # Initialization
    
    # Open and read the file
    with open('VideoProcessing/prediction.txt', 'r') as file:
        content = file.read()  
    
    # Pattern to match: "frame0171 have Pores [label1]: 175"
    pattern = r"frame(\d+) have Pores \[label1\]: (\d+)"
    matches = re.findall(pattern, content) # Find the pattern in the content 

    for match in matches:
        frame_number = int(match[0])
        pore_count = int(match[1])
        data.append((frame_number, pore_count))
        
    return data

def compare_pore_eda_timeseries(prediction_data, eda_file):
    # Extract pore data
    pore_frames, pore_counts = zip(*prediction_data)
    
    # Get EDA data
    eda_time, eda_values = get_EDA_data(eda_file)
    
    # Create two separate plots (no shared x-axis)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Plot 1: Pore Count Time Series with its own timeline
    ax1.plot(pore_frames, pore_counts, 'b-', linewidth=2, label='Pore Count', alpha=0.8)
    ax1.set_ylabel('Pore Count', fontsize=12, color='blue')
    ax1.set_xlabel('Frame Number', fontsize=12)
    ax1.set_title('Sweat Pore Activation Time Series', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot 2: EDA Time Series with its own timeline
    if eda_time is not None and eda_values is not None:
        ax2.plot(eda_time, eda_values, 'r-', linewidth=2, label='EDA Signal', alpha=0.8)
        ax2.set_ylabel('EDA Values', fontsize=12, color='red')
        ax2.set_xlabel('Time', fontsize=12)
        ax2.set_title('EDA Signal Time Series', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc='upper left')
    
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    return
 

if __name__ == "__main__":
    # Check current working directory
    # print("Current directory:", os.getcwd())
    # get_EDA_data()
    eda_file = 'EDA_R.xlsx'
    prediction_data = get_prediction_data()
    compare_pore_eda_timeseries(prediction_data, eda_file)
    