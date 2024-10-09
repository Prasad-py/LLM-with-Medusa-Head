import torch
import torch.nn as nn

class MedusaHead(nn.Module):
    def __init__(self, base_model, num_heads=3):
        super().__init__()
        self.base_model = base_model
        self.num_heads = num_heads
        # Create multiple linear layers for Medusa heads
        self.heads = nn.ModuleList([nn.Linear(base_model.config.hidden_size, base_model.config.vocab_size) for _ in range(num_heads)])

    def forward(self, hidden_states):
        # Get logits from the base model
        base_logits = self.base_model.lm_head(hidden_states)
        # Get logits from each Medusa head
        medusa_logits = [head(hidden_states) for head in self.heads]
        return base_logits, medusa_logits

def speculative_decoding(model, prompt, max_length):
    input_ids = model.tokenizer.encode(prompt, return_tensors="pt")
    
    for _ in range(max_length):
        with torch.no_grad():
            outputs = model(input_ids)
            hidden_states = outputs.last_hidden_state[:, -1:]
            base_logits, medusa_logits = model.medusa_head(hidden_states)
            
            # Combine predictions from all heads
            combined_logits = torch.stack([base_logits] + medusa_logits, dim=1)
            probs = torch.softmax(combined_logits, dim=-1)
            
            # Sample from the combined distribution
            next_token = torch.multinomial(probs.view(-1), 1).view(1, -1)
            
            # Append the new token to the input sequence
            input_ids = torch.cat([input_ids, next_token], dim=-1)
    
    return model.tokenizer.decode(input_ids[0])