import os
import dashscope

# The following URL is for the Singapore region. If you use a model in the Beijing region, replace the URL with: https://dashscope.aliyuncs.com/api/v1
dashscope.base_http_api_url = 'https://dashscope-intl.aliyuncs.com/api/v1'



def transcribe():
    # Replace ABSOLUTE_PATH/welcome.mp3 with the absolute path of your local audio file.
    audio_file_path = "/data/projects/13003558/zoux/datasets/samples/datasets_hf_stage_AudioLLM_v3/datasets_multimodal/train/ASR/MDT2020S033MinnanDialectScriptedSpeechCorpus_hok_30_ASR_31_context.wav"

    messages = [
        {
            "role": "system",
            "content": [
                # Configure the Context for customized recognition here.
                {"text": "this is"},
            ]
        },
        {
            "role": "user",
            "content": [
                {"audio": audio_file_path},
            ]
        }
    ]
    response = dashscope.MultiModalConversation.call(
        # The API keys for the Singapore and Beijing regions are different. To obtain an API key: https://www.alibabacloud.com/help/en/model-studio/get-api-key
        # If you have not configured the environment variable, replace the following line with your Model Studio API key: api_key = "sk-xxx"
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model="qwen3-asr-flash",
        messages=messages,
        result_format="message",
        asr_options={
            "language": "zh", # Optional. If the language of the audio is known, you can specify it with this parameter to improve recognition accuracy.
            "enable_lid":True,
            "enable_itn":True
        }
    )
    print(response)


if __name__ == "__main__":
    transcribe()

