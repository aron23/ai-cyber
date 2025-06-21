# Payload-Byte Dataset Information

## Overview
The Payload-Byte datasets contain processed and labeled network packet data for intrusion detection. The data includes both packet header features and payload bytes transformed into integer format.

## Available Datasets

### 1. UNSW-NB15
- **Download Link**: https://drive.google.com/drive/folders/1xuPB6VxQD70qvSH1YU69E2ICzLHJI2mO?usp=sharing
- **Format**: CSV
- **Description**: Modern network intrusion detection dataset from UNSW

### 2. CIC-IDS2017
- **Download Link**: https://drive.google.com/drive/folders/1sFBgQyvO2Wde8fwgZXTgVFXrD2W7v8WB?usp=sharing
- **Format**: CSV
- **Description**: Canadian Institute for Cybersecurity intrusion detection dataset

## Dataset Structure
Each dataset contains:
- **Packet Header Features** (4 features):
  - Source IP
  - Destination IP
  - Source Port
  - Destination Port
  
- **Payload Data** (1500 features):
  - Byte-wise representation (0-255 integer values)
  - Limited to 1500 bytes following de facto packet size limit
  - Transformed from hex values to integers per byte

## Download Instructions
1. Visit the Google Drive links above
2. Download all CSV files to this directory
3. Verify data integrity using the validation scripts

## Citation
If using this data, please cite:
```
@INPROCEEDINGS{Payload-Byte,
  author={Farrukh, Yasir Ali and Khan, Irfan and Wali, Syed and Bierbrauer, David and Pavlik, John A. and Bastian, Nathaniel D.},
  booktitle={2022 IEEE/ACM International Conference on Big Data Computing, Applications and Technologies (BDCAT)}, 
  title={Payload-Byte: A Tool for Extracting and Labeling Packet Capture Files of Modern Network Intrusion Detection Datasets}, 
  year={2022},
  pages={58-67},
  doi={10.1109/BDCAT56447.2022.00015}}
```