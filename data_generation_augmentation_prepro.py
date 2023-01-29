import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


############ Data Generation ###############
# Initialize empty dataframe with desired column names
columns = ["seconds_elapsed", "acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]
df = pd.DataFrame(columns=columns)
labels = []
data_list = []

# Set the current working directory to the 'data' folder
path = os.path.join('/Users/sdonzis/Desktop/Projet_IACHM/', 'classes')
if os.path.exists(path) and os.path.isdir(path):
    os.chdir(path)
else:
    print("The path does not exist or is not a directory")

# Iterate through folders
for classes in os.listdir():
    class_path = os.path.join(path, classes)
    if os.path.isdir(class_path):
        for file in os.listdir(class_path):
            if os.path.isdir(os.path.join(class_path, file)):                
                # Read accelerometer data
                accelerometer_file = os.path.join(class_path, file, "Accelerometer.csv")
                if os.path.exists(accelerometer_file) and os.path.isfile(accelerometer_file):
                    accelerometer_data = pd.read_csv(accelerometer_file)
                    accelerometer_data = accelerometer_data[["seconds_elapsed", "x", "y", "z"]]
                    accelerometer_data.columns = ["seconds_elapsed", "acc_x", "acc_y", "acc_z"]
                else:
                    continue

                # Read gyroscope data
                gyroscope_file = os.path.join(class_path, file, "Gyroscope.csv")
                if os.path.exists(gyroscope_file) and os.path.isfile(gyroscope_file):
                    gyroscope_data = pd.read_csv(gyroscope_file)
                    gyroscope_data = gyroscope_data[["x", "y", "z"]]
                    gyroscope_data.columns = ["gyro_x", "gyro_y", "gyro_z"]
                else:
                    continue
                
                sample_array = np.transpose(np.array([accelerometer_data["acc_x"].values,
                                             accelerometer_data["acc_y"].values,
                                             accelerometer_data["acc_z"].values,
                                             gyroscope_data["gyro_x"].values,
                                             gyroscope_data["gyro_y"].values,
                                             gyroscope_data["gyro_z"].values,
                                            ]))

                # append data list
                data_list.append(sample_array)
                
                # Update labels list
                labels.append(int(classes))
                
# convert to array
data = np.array(data_list)

# check imported data shape
print(data.shape)

############ Data Augmentation ###############
def data_augmentation(data, labels, num_augmentations):
    # Create an empty list for augmented data
    augmented_time_series_list = []
    augmented_labels = []
    index_augmented = []
    
    j=0
    for i in range(num_augmentations): 
        for sample, label in zip(data, labels):
            # Copy the original time series to start with
            augmented_ts = sample.copy()
        
            do_flip, do_noise, do_scale, do_shift = False,False,False,False
                
            # Scaling data
            do_scale = np.random.choice([True, False])
            if do_scale:
                # Choose a random scaling factor
                scale_factor = np.random.uniform(low=0.5, high=2)
                # Scale the time series
                augmented_ts = augmented_ts * scale_factor
    
            # Shifting data
#             do_shift = np.random.choice([True, False])
#             if do_shift:
            # Choose a random shift value
            shift_steps = np.random.randint(low=-10, high=10)
            for feature in range(augmented_ts.shape[1]):
                # Shift the time series for the individual feature
                augmented_ts[:, feature] = np.roll(augmented_ts[:, feature], shift_steps)
    
            # Adding noise
            do_noise = np.random.choice([True, False])
            if do_noise:
                # Get the maximum value of each feature
                max_values = np.max(augmented_ts, axis=0)
                # Calculate the noise mean for each feature using the maximum value
                noise_mean = np.random.uniform(low=-0.1, high=0.1, size=augmented_ts.shape[1])*max_values
                noise_std = np.random.uniform(low=0.2, high=0.3)
                # Add noise to the time series
                noise = np.random.normal(0, noise_std, augmented_ts.shape)
                augmented_ts = augmented_ts + noise
            
            # Rotating data
            do_flip = np.random.choice([True, False])
            if do_flip:
                # Flip the data
                augmented_ts = -1*augmented_ts
    
            if not do_flip and not do_noise and not do_scale and not do_shift:
                # Update augmented indexes
                index_augmented.append(j)
                                
            # Append the augmented time series to the list         
            augmented_time_series_list.append(augmented_ts)
        
            # Update label data
            augmented_labels.append(label)
                
            j+=1
                
    # Convert to numpy array
    augmented_data = np.array(augmented_time_series_list)
    augmented_labels = np.array(augmented_labels)
    
    return augmented_data, augmented_labels, index_augmented

# Define the number of augmentations to perform
num_augmentations = 8

augmented_data, augmented_labels, index_augmented = data_augmentation(data, labels, num_augmentations)

# check number of new data
print(len(augmented_data),len(augmented_labels))

################## Ploting #####################

# Number of features
num_features = 6
index = 0         # index of original sample to plot
n_augmented = 2   # number of augmentated samples

# Plot original sample in red
fig, axs = plt.subplots(2, 3, figsize=(15, 10))
axs = axs.ravel()
for i in range(num_features):
    axs[i].plot(data[index][:, i], 'r', linewidth=2)
    axs[i].set_title(str(columns[i+1]))

# Plot augmented samples in blue
for i in range(0, min(data.shape[0]*n_augmented,int(len(augmented_data)/n_augmented)), index+data.shape[0]):
# for i in [150]:
    for j in range(num_features):
        axs[j].plot(augmented_data[i][:, j], 'b', linewidth=0.7, alpha=0.7)

plt.show()


################## Preprocessing #####################
# delete duplicates
augmented_data_ = np.delete(augmented_data, index_augmented, axis=0)
augmented_labels_ = np.delete(augmented_labels, index_augmented, axis=0)

# Concatenate original dataset + augmented
data_original_augmented = np.concatenate((data, augmented_data_))

data_list=[]

for sample in data_original_augmented:
    
    sample_shape = sample_array.shape[0]

    #Padding and truncating dataset
    max_length = 64
    
    if sample_shape > max_length:
        sample_array = sample_array[:max_length]
    else:
        zero_array = np.ones((max_length-sample_shape, 6))*sample_array[-1]
        sample_array = np.vstack((sample_array, zero_array))

    # Append data list
    data_list.append(sample_array)
        
processed_data_sequence_augmented = np.array(data_list)
data_labels = np.array(augmented_labels_)

print("Processed Augmented Data Shape:", processed_data_sequence_augmented.shape)

np.save("processed_data_sequence.npy", processed_data_sequence_augmented)
np.save("data_labels.npy",data_labels)