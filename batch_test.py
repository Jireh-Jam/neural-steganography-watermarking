"""
batch_test.py - Advanced steganography evaluation with comprehensive metrics
"""

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from models import encoder, decoder
from utils import load_image, text_to_bits, bits_to_text

def calculate_ber(original_bits, decoded_bits):
    """Calculate Bit Error Rate with alignment"""
    min_len = min(len(original_bits), len(decoded_bits))
    return np.mean(original_bits[:min_len] != decoded_bits[:min_len])

def calculate_entropy(image):
    """Calculate Shannon entropy of an image"""
    hist = np.histogram(image, bins=256, range=(0,1))[0]
    hist = hist[hist > 0] / hist.sum()
    return -np.sum(hist * np.log2(hist))

def batch_test(image_paths, secret_message, max_images=None):
    """
    Enhanced batch testing with additional metrics
    
    Args:
        image_paths: List of image paths
        secret_message: Secret text to embed
        max_images: Maximum number of images to process
    
    Returns:
        Dictionary with metrics and visualization data
    """
    metrics = {
        # Image Quality
        'psnr': [],
        'ssim': [],
        'ms_ssim': [],
        'entropy_diff': [],
        
        # Message Fidelity
        'bit_accuracy': [],
        'ber': [],
        'exact_matches': 0,
        'char_error_rate': [],
        
        # Performance
        'encode_time': [],
        'decode_time': [],
        
        # Statistical
        'failed': 0,
        'per_image': []
    }

    for img_path in tqdm(image_paths[:max_images] if max_images else image_paths):
        try:
            # Timing
            start_time = time.time()
            
            # Test current image
            test_result = test_embed_extract(
                encoder=encoder,
                decoder=decoder,
                image_path=img_path,
                message=secret_message
            )
            
            if test_result is None:
                metrics['failed'] += 1
                continue
            
            encode_time = time.time() - start_time
            decode_start = time.time()
            
            # Extract metrics
            original = test_result['original_image']
            stego = test_result['stego_image']
            
            # Remove batch dim if needed
            if len(original.shape) == 4: original = original[0]
            if len(stego.shape) == 4: stego = stego[0]
            
            # Image Quality Metrics
            current_psnr = psnr(original, stego)
            current_ssim = ssim(original, stego, multichannel=True)
            entropy_diff = calculate_entropy(stego) - calculate_entropy(original)
            
            # Message Metrics
            original_bits = text_to_bits(secret_message).flatten()
            decoded_bits = (text_to_bits(test_result['recovered_message'])).flatten()
            bit_accuracy = np.mean(original_bits == decoded_bits[:len(original_bits)])
            current_ber = calculate_ber(original_bits, decoded_bits)
            
            # Character-level metrics
            char_errors = sum(1 for a,b in zip(secret_message, test_result['recovered_message']) if a != b)
            char_error_rate = char_errors / len(secret_message)
            
            decode_time = time.time() - decode_start
            
            # Update metrics
            metrics['psnr'].append(current_psnr)
            metrics['ssim'].append(current_ssim)
            metrics['entropy_diff'].append(entropy_diff)
            metrics['bit_accuracy'].append(bit_accuracy)
            metrics['ber'].append(current_ber)
            metrics['char_error_rate'].append(char_error_rate)
            metrics['encode_time'].append(encode_time)
            metrics['decode_time'].append(decode_time)
            
            if secret_message == test_result['recovered_message']:
                metrics['exact_matches'] += 1
                
            # Store per-image details
            metrics['per_image'].append({
                'path': img_path,
                'psnr': current_psnr,
                'ssim': current_ssim,
                'ber': current_ber,
                'encode_time': encode_time,
                'decode_time': decode_time,
                'exact_match': secret_message == test_result['recovered_message'],
                'recovered_message': test_result['recovered_message']
            })
            
        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")
            metrics['failed'] += 1
    
    # Calculate aggregates
    for key in ['psnr', 'ssim', 'ber', 'encode_time', 'decode_time', 'bit_accuracy']:
        if metrics[key]:
            metrics[f'avg_{key}'] = np.mean(metrics[key])
            metrics[f'std_{key}'] = np.std(metrics[key])
        else:
            metrics.update({f'avg_{key}': 0, f'std_{key}': 0})
    
    metrics['success_rate'] = len(metrics['per_image']) / (len(metrics['per_image']) + metrics['failed'])
    
    return metrics

def visualize_results(metrics):
    """Generate comprehensive visualizations"""
    plt.figure(figsize=(18, 10))
    
    # 1. Quality Metrics
    plt.subplot(2, 3, 1)
    plt.hist(metrics['psnr'], bins=20, alpha=0.7)
    plt.title(f"PSNR Distribution\nAvg: {metrics['avg_psnr']:.2f} ± {metrics['std_psnr']:.2f} dB")
    plt.xlabel('PSNR (dB)')
    
    plt.subplot(2, 3, 2)
    plt.hist(metrics['ssim'], bins=20, alpha=0.7)
    plt.title(f"SSIM Distribution\nAvg: {metrics['avg_ssim']:.4f}")
    plt.xlabel('SSIM')
    
    # 2. Message Accuracy
    plt.subplot(2, 3, 3)
    plt.hist(metrics['ber'], bins=20, alpha=0.7, color='red')
    plt.title(f"Bit Error Rate\nAvg: {metrics['avg_ber']:.2%}")
    plt.xlabel('BER')
    
    # 3. Performance
    plt.subplot(2, 3, 4)
    plt.scatter(metrics['encode_time'], metrics['decode_time'], alpha=0.6)
    plt.title("Encoding vs Decoding Time")
    plt.xlabel('Encode Time (s)')
    plt.ylabel('Decode Time (s)')
    
    # 4. Error Analysis
    plt.subplot(2, 3, 5)
    error_types = ['Exact Matches', 'Char Errors', 'Bit Errors']
    values = [
        metrics['exact_matches'],
        sum(m['exact_match'] is False for m in metrics['per_image']),
        sum(ber > 0 for ber in metrics['ber'])
    ]
    plt.bar(error_types, values, color=['green', 'orange', 'red'])
    plt.title("Error Type Distribution")
    
    plt.tight_layout()
    plt.savefig('steganography_metrics.png')
    plt.close()

def generate_report(metrics):
    """Create a markdown report"""
    report = f"""# Steganography System Evaluation Report

## Summary Statistics
- **Images Processed**: {len(metrics['per_image']) + metrics['failed']}
- **Success Rate**: {metrics['success_rate']:.2%}
- **Exact Matches**: {metrics['exact_matches']} ({metrics['exact_matches']/len(metrics['per_image']):.2%})

## Quality Metrics
| Metric | Average | Std Dev |
|--------|---------|---------|
| PSNR (dB) | {metrics['avg_psnr']:.2f} | {metrics['std_psnr']:.2f} |
| SSIM | {metrics['avg_ssim']:.4f} | {metrics['std_ssim']:.4f} |
| Bit Error Rate | {metrics['avg_ber']:.2%} | {metrics['std_ber']:.2%} |

## Performance
- **Avg Encode Time**: {metrics['avg_encode_time']:.4f}s ± {metrics['std_encode_time']:.4f}
- **Avg Decode Time**: {metrics['avg_decode_time']:.4f}s ± {metrics['std_decode_time']:.4f}

![Metrics Visualization](steganography_metrics.png)
"""
    with open('steganography_report.md', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    import glob
    import time
    import pandas as pd
    
    # Configuration
    test_images = glob.glob("data/test/*.jpg")[:1000]  # First 1000 images
    secret_msg = "MySecretKey123"
    
    # Run evaluation
    print("Starting comprehensive evaluation...")
    start_time = time.time()
    metrics = batch_test(test_images, secret_msg)
    print(f"Completed in {time.time()-start_time:.2f} seconds")
    
    # Generate outputs
    visualize_results(metrics)
    generate_report(metrics)
    
    # Save detailed results
    pd.DataFrame(metrics['per_image']).to_csv('detailed_results.csv', index=False)
    print("Report generated: steganography_report.md")
    print("Detailed results: detailed_results.csv")
