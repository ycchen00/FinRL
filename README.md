Based on [AI4Finance-Foundation/FinRL: A Deep Reinforcement Learning Framework for Automated Trading in Quantitative Finance. NeurIPS 2020. 🔥](https://github.com/AI4Finance-Foundation/FinRL), severl large adoptions were made:
- Proper rewards for train, more parameters with piecewise function, considering time penality
- More complicated states
- Training procedures: States change real-time based on trading
- Trading patterns: More in, More out. Rewards and states are updated in pairs
