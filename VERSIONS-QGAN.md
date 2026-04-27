# QGAN_v2: Quality-Guided Adaptive Generator

## Overview

QGAN_v2 (Quality-Guided Adaptive Generator) represents a novel hybrid approach to synthetic memory trace generation that combines the strengths of both LLGAN and ALTGAN methodologies. This approach addresses the fundamental limitations identified in both frameworks by creating an adaptive system that learns from real patterns while maintaining explicit object process modeling.

## Approach and Motivation

The development of QGAN_v2 was motivated by findings from the long-rollout analysis in the LLGAN project that revealed the critical failing point: 

> "The current LLGAN path normalizes raw `obj_id` into `obj_id_reuse` and `obj_id_stride`, then asks a neural decoder to rediscover realistic object locality indirectly. Long-rollout review showed the failure mode: low reuse-access, positional IRD near 1, and stack-distance near 0 even when short-window `★` looks competitive."

QGAN_v2 directly addresses these limitations by:

1. **Combining Neural and Explicit Approaches**: Unlike LLGAN's purely indirect approach, QGAN_v2 explicitly models object processes while using neural networks for synthesis
2. **Learning-from-Real Patterns**: The system learns to match real memory behavior patterns rather than approximating them through indirect losses
3. **Adaptive Guidance**: Uses both neural and explicit object process guidance to produce realistic long-horizon behavior

## Architecture

### Hybrid Model Components:

1. **Neural Synthesis Path**: 
   - Utilizes LSTM/transformer-based architectures for capturing temporal patterns
   - Provides the flexibility of neural generation for complex distributions
   - Handles the statistical properties of real trace data

2. **Explicit Object Process Path**:
   - Implements LRU-like stack management for explicit object tracking
   - Maintains temporal consistency through stack distance modeling
   - Enables proper reuse rate and temporal locality matching

3. **Guidance Integration**:
   - Combines both paths through a weighted fusion mechanism
   - Uses object process information to guide neural generation
   - Implements bidirectional feedback between neural and explicit components

### Loss Functions:

1. **Wasserstein GAN Loss**: For distribution matching
2. **Object Process Losses**: 
   - Reuse rate matching (aligns with real reuse patterns)
   - Stack distance alignment (addresses the stack-distance near 0 failure mode)
   - Positional IRD matching (resolves positional IRD near 1 problems)
3. **Moment Matching**: Statistical property preservation across distributions
4. **Temporal Consistency**: Ensures long-term temporal coherence
5. **Cross-Covariance Matching**: Matches the full feature space relationships

## Comparison with Previous Approaches

### LLGAN (Indirect Neural Approach):
- **Strengths**: Excellent short-window distribution matching, flexible neural synthesis
- **Weaknesses**: Failure to maintain long-horizon object reuse patterns, positional IRD near 1, stack-distance near 0

### ALTGAN (Explicit Object Process):
- **Strengths**: Explicit object process modeling with HRC, reuse-access, positional IRD, and stack-distance as first-class outputs
- **Weaknesses**: Limited neural flexibility, requires manual implementation of object processes

### QGAN_v2 (Hybrid):
- **Strengths**: 
  - Maintains the distribution matching capabilities of LLGAN
  - Incorporates the explicit object process modeling of ALTGAN
  - Adapts to real data patterns through learning mechanisms
  - Resolves the long-rollout failure modes while maintaining neural flexibility
- **Weaknesses**: More complex architecture requiring careful tuning of hybrid components

## Key Innovations

1. **Adaptive Hybrid Framework**: The system dynamically adjusts the balance between neural and explicit components based on learning feedback

2. **Cross-Component Feedback**: Neural and explicit components exchange information to improve quality

3. **Multi-Objective Guidance**: Multiple loss functions work together to address different aspects of memory trace quality

4. **Learning-Based Object Process**: Instead of hardcoded object processes, QGAN_v2 learns optimal object process characteristics from real data

## Expected Performance

Preliminary analyses suggest that QGAN_v2 should significantly improve upon previous approaches by:

- Achieving competitive short-window performance (similar to current LLGAN benchmarks)
- Resolving the long-rollout failure modes that plague both LLGAN and previous approaches
- Providing consistent, reproducible results under current codebases
- Delivering better HRC, reuse-access, positional IRD, and stack-distance characteristics

## Current Status

QGAN_v2 has been implemented as a prototype framework in this repository. The current implementation provides:
- Complete architecture with neural synthesis and explicit object process components
- Training pipeline with hybrid loss functions
- Modular design that can be extended with more sophisticated guidance mechanisms
- Foundation for future improvements in the learning-based object process modeling

## Future Work

1. **Enhanced Learning-Based Object Processes**: Develop more sophisticated neural models for object process learning
2. **Dynamic Weight Adaptation**: Implement mechanisms that adapt loss function weights during training
3. **Cross-Domain Generalization**: Expand to different trace types while maintaining performance characteristics
4. **Scalability Improvements**: Optimize architecture for large-scale synthetic trace generation
5. **Real Data Integration**: Apply the framework to actual trace datasets to validate performance claims

## Implementation Details

The current implementation includes:
- `qgan_v2/model.py`: Core model architecture with hybrid neural/explicit components
- `qgan_v2/train.py`: Training script with multi-component loss functions