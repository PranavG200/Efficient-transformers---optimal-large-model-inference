# Efficient Transformers : Post Training Pruning and Quantization for optimal large model inference
Enhancing transformer inference by employing post-training pruning (PTP) and quantization (PTQ). We explore how PT inference optimization methods can reduce computational complexity, memory footprint &amp; inference latency and make things run faster while using less memory. We conducted experiments using a BERT model fine-tuned on the MRPC dataset and tested it on Question-Answering Pair (QQP) for pruning results. The implementation is based on PyTorch and the HuggingFace Transformers library. Below is an overview of our findings and the key metrics analyzed.

## Key Metrics

### Accuracy Analysis

#### Pruning vs Accuracy

| Pruning Level | Accuracy (Baseline) | Accuracy (Optimized) |
|---------------|---------------------|----------------------|
| 0%            | 89.25%              | 89.25%               |
| 14%           | 88.5%               | 87.7%                |
| 30%           | 82.1%               | 79.8%                |

#### Quantization vs Accuracy

| Quantization Level | Accuracy (Baseline) | Accuracy (Optimized) |
|--------------------|---------------------|----------------------|
| No Quantization    | 89.25%              | 89.25%               |
| 8-bit              | 82%                 | 80.1%                |
| 16-bit             | 88.9%               | 86.5%                |

#### Pruning + Quantization vs Accuracy

| Pruning and Quantization Level | Accuracy (Baseline) | Accuracy (Optimized) |
|--------------------------------|---------------------|----------------------|
| None                           | 89.25%              | 89.25%               |
| 14% Pruning + 16-bit           | 86.8%               | 84.6%                |

### Inference Time and Speedup Analysis

| Method                      | Time taken (Baseline) | Time taken (Optimized) |
|-----------------------------|-----------------------|------------------------|
| None                        | 130s                  | 130s                   |
| Pruning (22%) Only          | 93s                   | 80s                    |
| Quantization Only (16 Bit)  | 91s                   | 93s                    |
| Pruning + Quantization      | 105s                  | 98s                    |

### Model Size Analysis

| Method                      | Model Size (Baseline) | Model Size (Optimized) |
|-----------------------------|-----------------------|------------------------|
| None                        | 438 Mb                | 438 Mb                 |
| Pruning (22%) Only          | 290 Mb                | 278 Mb                 |
| Quantization Only (16 Bit)  | 230 Mb                | 242 Mb                 |
| Pruning + Quantization      | 155 Mb                | 171 Mb                 |

## Methodology

Our methodology involved fine-tuning a BERT model on the MRPC dataset and evaluating it on QQP. We then applied pruning and quantization techniques to optimize the model for inference. The key focus was on accuracy, model size, and inference time, which are crucial metrics for assessing the efficiency and efficacy of the optimized models.

## Comparison with Intel's Neural Compressor

We compared our results with Intel's Neural Compressor, showcasing the effectiveness of our approach in achieving similar model sizes and improved accuracy metrics.

## Running the Code

To replicate our experiments and results, follow these steps:

1. Clone this repository.
2. Install the required dependencies.
3. Run the provided scripts for data preprocessing, model fine-tuning, and optimization techniques.
4. Evaluate the performance metrics based on the generated results.

## Conclusion

Our optimization techniques, including pruning and quantization, demonstrate significant improvements in model size and inference speed while maintaining competitive accuracy levels. This project provides valuable insights into optimizing transformer models for efficient and effective inference in real-world scenarios.
