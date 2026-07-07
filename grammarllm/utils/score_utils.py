import torch

def print_first_token_probs(generation_output, top_k=10):
    """
    Demonstrates how to access first token probabilities from the generation output.
    Args:
        generation_output: The output from generate_text.
        top_k: Number of top tokens to display. If None, considers all (vocab size).
    """
    # Check if output is a list (multiple sequences) or dict (single sequence)
    if isinstance(generation_output, dict):
        generation_output = [generation_output]
        
    for i, result in enumerate(generation_output):
        print(f"\n--- Sequence {i+1} ---")
        print(f"Text: {result['text']}")
        
        # Access logits for the first generated token (step 0)
        # result['scores'] is a list of steps. result['scores'][0] is the logits list for step 0.
        first_step_filtered_logits = result['scores'][0]
        first_step_original_logits = result['original_scores'][0]
        
        # Convert to tensor for softmax
        # Note: These are lists, so we wrap them in torch.tensor
        filt_tensor = torch.tensor(first_step_filtered_logits)
        orig_tensor = torch.tensor(first_step_original_logits)
        
        # Calculate probabilities
        filtered_probs = torch.softmax(filt_tensor, dim=-1)
        original_probs = torch.softmax(orig_tensor, dim=-1)
        
        # Determine k
        k = top_k if top_k is not None else filtered_probs.size(-1)
        
        # Get top k
        top_filt_val, top_filt_idx = torch.topk(filtered_probs, k)
        top_orig_val, top_orig_idx = torch.topk(original_probs, k)
        
        print(top_filt_idx)
        print(top_orig_idx)
        print(top_filt_val)
        print(top_orig_val)

        print("\n\n")
        
        print(f"Top {k} First Token Probabilities:")
        
        # Print side-by-side comparison for top k
        print(f"{'Rank':<5} | {'Original ID':<12} | {'Orig Prob':<10} | {'Filtered ID':<12} | {'Filt Prob':<10}")
        print("-" * 65)
        
        display_limit = k if (top_k is not None and top_k < 50) else 10
        if top_k is None:
            print("(Displaying top 10 of full distribution)")
        
        for r in range(display_limit):
             o_id = top_orig_idx[r].item()
             o_p = top_orig_val[r].item()
             f_id = top_filt_idx[r].item()
             f_p = top_filt_val[r].item()
             print(f"{r+1:<5} | {o_id:<12} | {o_p:.4f}     | {f_id:<12} | {f_p:.4f}")
             
        if top_k is None or (top_k is not None and top_k > display_limit):
            print(f"... (and {k - display_limit} more)")
