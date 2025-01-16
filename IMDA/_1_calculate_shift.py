import json
from multiprocessing import Pool
import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
from tqdm import tqdm


def find_transition_position(sequence):
    # Convert the sequence to a numpy array for easier manipulation
    sequence = np.array(sequence)
    
    # Calculate the mean of the entire sequence
    mean_value = np.mean(sequence)
    
    # Compute the cumulative sum of deviations from the mean
    cumulative_sum = np.cumsum(sequence - mean_value)
    
    # Find the index where the cumulative sum reaches its maximum or minimum
    transition_index = np.argmax(np.abs(cumulative_sum))
    
    return transition_index+1


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

    return list(indexes), list(filtered_numbers)


def calculate_offset_without_outliers(diff, segments_filepath):

    offsets = diff["offset"]
    numbers = offsets
    filtered_2_number_indexes, filtered_numbers = remove_outliers(numbers)

    # Generate or load your sequence of numbers
    data = np.array(filtered_numbers)

    # Use Gaussian KDE to estimate the density
    kde = gaussian_kde(data, bw_method=0.2)
    x = np.linspace(min(data), max(data), 1000)
    kde_values = kde(x)*50

    # Identify the most and second most dense regions
    peaks, _ = find_peaks(kde_values)
    breakpoint()
    sorted_peaks = peaks[np.argsort(kde_values[peaks])][-3:]  # Get the indices of the two highest peaks
    breakpoint()
    first_peak = x[sorted_peaks[-1]]
    breakpoint()
    if len(sorted_peaks) == 2:
        second_peak = x[sorted_peaks[-2]]
        third_peak = second_peak
    if len(sorted_peaks) == 3:
        second_peak = x[sorted_peaks[-2]]
        third_peak = x[sorted_peaks[-3]]
    else:
        second_peak = first_peak
        third_peak = first_peak
    breakpoint()
    
    # span=abs(first_peak-second_peak)/3
    # first_peak_amount   = len([num for num in filtered_numbers if first_peak-span <= num <= first_peak+span])
    # second_peak_amount  = len([num for num in filtered_numbers if second_peak-span <= num <= second_peak+span])
    # if first_peak_amount > second_peak_amount * 3 or abs(first_peak-second_peak) < 1:
    #     numbers = [num for num in filtered_numbers if first_peak-0.3 <= num <= first_peak+0.3]
    #     if numbers:
    #         only_peak = np.mean([num for num in filtered_numbers if first_peak-0.3 <= num <= first_peak+0.3])
    #     else:
    #         only_peak = first_peak
    #     first_peak  = only_peak
    #     second_peak = only_peak


    # Plot the KDE and cluster centers
    plt.hist(filtered_numbers, bins=100)
    plt.plot(x, kde_values, color='r')
    plt.axvline(first_peak, color='b', linestyle='dashed', linewidth=4, label='Most Dense Region')
    plt.axvline(second_peak, color='r', linestyle='dotted', linewidth=4, label='Second Dense Region')
    plt.axvline(third_peak, color='g', linestyle='dotted', linewidth=4, label='Third Dense Region')
    plt.savefig(segments_filepath.replace(".ctm", ".png")) 
    # plt.savefig("/scratch/users/astar/ares/zoux/workspaces/data_process/test.png") 
    plt.clf()



    plt.plot(filtered_numbers, label='diff')
    plt.ylim(bottom=min(-10, min(filtered_numbers)), top=max(10, max(filtered_numbers)))
    plt.savefig(segments_filepath.replace(".ctm", "_diff.png"))
    plt.clf()

    transition_index_in_filtered_numbers = find_transition_position(filtered_numbers)
    transition_index_in_numbers          = filtered_2_number_indexes[transition_index_in_filtered_numbers]
    transition_index_in_transcriptions   = diff["index"][transition_index_in_numbers]

    return max(first_peak, second_peak), min(first_peak, second_peak), transition_index_in_transcriptions


def generate_txt_png(args):
    part, item = args
    
    audio_filepath    = item["audio_filepath"]
    segments_filename = audio_filepath.split("/")[-1].replace(".wav", ".ctm")
    segments_filepath = os.path.join(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/NFA_output/ctm/segments", segments_filename)
    segments          = open(segments_filepath).readlines()
    transcriptions    = item["transcriptions"]

    assert len(segments) == len(transcriptions), f"length not match: {segments_filename} {len(segments)} {len(transcriptions)}"

    diff = {"index":[], "offset":[]} 
    with open(segments_filepath.replace(".ctm", ".txt"), "w") as f:
        for i, segment in enumerate(segments):
            segment        = segment.strip().split()
            start          = float(segment[2])
            end            = start+float(segment[3])
            sentence       = segment[4].replace("<space>", " ")
            offset_start   = transcriptions[i]["start"]-start
            offset_end     = transcriptions[i]["end"]-end
            offset_average = (offset_start + offset_end)/2
            f.write("{:.2f} || {:.2f} || {:.2f} || {} || {}\n".format(offset_start, offset_end, offset_average, transcriptions[i]["sentence"], sentence))
            if len(sentence.split()) < 2:
                continue

            diff["index"].append(i)
            diff["offset"].append(offset_start)
        center1, center2, transition_index_in_transcriptions = calculate_offset_without_outliers(diff, segments_filepath)
        f.write("{:.2f} || {:.2f} || {}\n".format(center1, center2, transition_index_in_transcriptions))
    item["center1"] = center1
    item["center2"] = center2
    item["transition_index"] = transition_index_in_transcriptions
    return json.dumps(item, ensure_ascii=False)


# for part in ["PART6", "PART5", "PART4", "PART3"]:
for part in ["PART4"]:

    print(f"start {part}", flush=True)
    lines = open(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/manifest_with_transcriptions.jsonl").readlines()
    results=[generate_txt_png((part, json.loads(lines[0])))]

    # with Pool(processes=32) as pool:
    #     params = [(part, json.loads(line)) for line in lines]
    #     results = list(tqdm(pool.imap_unordered(generate_txt_png, params), total=len(params)))

    with open(f"/scratch/users/astar/ares/zoux/workspaces/data_process/_data_in_processing/imda/imda_raw/{part}/manifest_with_transcriptions_and_shift.jsonl", "w", encoding="utf-8") as f:
        for result in results:
            f.write(result+"\n")

    print(f"complete {part}", flush=True)

