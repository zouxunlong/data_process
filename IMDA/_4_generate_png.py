import json
from multiprocessing import Pool
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from tqdm import tqdm


def remove_outliers(numbers):
    if not numbers:
        return []

    # Step 1: Sort the list of numbers
    sorted_numbers = sorted(numbers)

    # Step 2: Calculate Q1 and Q3
    n = len(sorted_numbers)
    Q1 = sorted_numbers[n // 4]
    Q3 = sorted_numbers[(3 * n) // 4]

    # Step 3: Compute the IQR
    IQR = Q3 - Q1

    # Step 4: Determine the lower and upper bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Step 5: Filter out the outliers
    filtered_numbers_with_index = [(i, num) for i, num in enumerate(numbers) if lower_bound <= num <= upper_bound]
    
    indexes, filtered_numbers=zip(*filtered_numbers_with_index)

    return list(filtered_numbers)


def generate_png(part, diff, png_filepath):

    offset_average = diff["offset_average"]

    filtered_data_average = remove_outliers(offset_average)

    # Generate or load your sequence of numbers
    filtered_data_average = np.array(filtered_data_average)
    
    # Use Gaussian KDE to estimate the density
    try:
        kde = gaussian_kde(filtered_data_average, bw_method=0.2)
        x = np.linspace(min(filtered_data_average), max(filtered_data_average), 1000)
        kde_values = kde(x)*10
    except:
        print(f"error: {part} {png_filepath}")
        return 0, 0

    # Identify the most and second most dense regions
    peaks, _ = find_peaks(kde_values)
    sorted_peaks = peaks[np.argsort(kde_values[peaks])][-1:]  # Get the indices of the two highest peaks
    first_peak = x[sorted_peaks[-1]]


    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # # Plot the KDE and cluster centers on the first subplot
    # ax1.hist(filtered_data_average, bins=100)
    # ax1.plot(x, kde_values, color='r')
    # ax1.axvline(first_peak, color='b', linestyle='dashed', linewidth=4, label='Most Dense Region')

    # # Plot the average offset on the second subplot
    # ax2.plot(filtered_data_average, label='avg_offset')
    # ax2.set_ylim(bottom=min(-10, min(filtered_data_average)), top=max(10, max(filtered_data_average)))

    # # Save the figure with both subplots
    # plt.savefig(png_filepath)
    # plt.clf()
    # plt.close()

    mean_value     = np.mean(filtered_data_average)
    squared_errors = np.square(filtered_data_average - mean_value)
    mse            = np.mean(squared_errors)

    return first_peak, mse


def generate_txt_png(args):
    part, item = args

    audio_filepath    = item["audio_filepath"]
    segments_filename = audio_filepath.split("/")[-1].replace(".wav", ".ctm")
    segments_filepath = os.path.join(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/NFA_output/ctm/segments", segments_filename)
    txt_filepath      = os.path.join(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/txt_all", segments_filename.replace(".ctm", ".txt"))
    png_filepath      = os.path.join(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/png_all", segments_filename.replace(".ctm", ".png"))
    segments          = open(segments_filepath).readlines()
    transcriptions    = item["transcriptions"]

    assert len(segments) == len(transcriptions), f"length not match: {segments_filename} {len(segments)} {len(transcriptions)}"

    diff = {"index":[], "offset_start":[], "offset_end":[], "offset_average":[]} 
    with open(txt_filepath, "w") as f:
        for i, segment in enumerate(segments):
            segment        = segment.strip().split()
            start          = float(segment[2])
            end            = start+float(segment[3])
            sentence       = segment[4].replace("<space>", " ")
            offset_start   = transcriptions[i]["start"] - start
            offset_end     = transcriptions[i]["end"] - end
            offset_average = (offset_start + offset_end)/2
            f.write("{:.2f} || {:.2f} || {:.2f} || {} || {}\n".format(offset_start, offset_end, offset_average, transcriptions[i]["sentence"], sentence))
            if len(sentence.split()) < 2:
                continue

            diff["index"].append(i)
            diff["offset_start"].append(offset_start)
            diff["offset_end"].append(offset_end)
            diff["offset_average"].append(offset_average)

        first_peak, mse = generate_png(part, diff, png_filepath)
        f.write(f"{first_peak} || {mse}\n")

    item["peak"] = first_peak
    item["mse"] = mse

    return json.dumps(item, ensure_ascii=False)


for part in ["PART3", "PART4", "PART5","PART6"]:
    
    os.makedirs(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/txt_all", exist_ok=True)
    os.makedirs(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/png_all", exist_ok=True)

    print(f"start {part}", flush=True)
    lines = open(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/manifest.jsonl").readlines()

    with Pool(processes=16) as pool:
        params = [(part, json.loads(line)) for line in lines]
        results = list(tqdm(pool.imap_unordered(generate_txt_png, params), total=len(params)))

    print(f"complete {part}", flush=True)

