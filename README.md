# EARL
Environment Aware Reinforcement Learning


### Insight:

According to [Thousand Brains Theory](https://www.amazon.com/Thousand-Brains-New-Theory-Intelligence/dp/1541675819), by Jeff Hawkins, Humans constantly make predictions about the world they interact in. However, prediction is an intermediate step, and not the end goal. The end goal, loosely put is to update the "world-view" of the agent. This is an interesting idea because in standard Reinforcement Learning, the end goal of the agent is to make prediction given some state input that represents an abstraction of the world. These so called RL-agents are not **aware** of the world they interact in. Thus, these predictions seem quite sparse. This idea can be analogous to that of [task-based learning](https://arxiv.org/abs/1703.04529) where designing a direct end-to-end pipeline may lead to suboptimal performance as compared to breaking the problem into intermediate optimization tasks (while keeping track of meta optimization). 



### Idea:

The idea is as follows: instead of simply training an RL agent like DQN or actor-critic models to make prediction, why not have these models learn the mapping of the environment itself? If an agent understands the behavior of the environment, then predicting certain action to move in a particular direction is a trivial task. The key idea is to have 2 such models, one for predicting actions (call this model `f`), and the other for predicting next state given `f` (call this model `E`). The goal is to reduce the deviation from `E` prediction of next state, ![image](https://user-images.githubusercontent.com/43754306/129492398-269389e9-fcfd-49d6-866a-bbae357fb0c9.png) to the true next state, ![image](https://user-images.githubusercontent.com/43754306/129492404-f2436693-b146-4dde-9efd-478f685532b5.png). Here's what it loks like mathematically: ![image](https://user-images.githubusercontent.com/43754306/129492547-bb21173d-6e64-4564-a18e-0f80691768be.png).

Theta is shared across prediction and environment models to be able to optimize prediction module. The key question now remains, do we learn the environment from scratch or start from current/initial copy of the environment and only model the deviations of the action instead of true envrionment behavior? Starting from scratch seems more relaistic in the sense that babies have very little understanding of the world. However, that feels naive because babies do have _some_ idea of the world through DNA and other models inherited via evolution. On the other hand, handing a copy of the environment and only tasked with modeling deviations feels like cheaing. So, something in between should be more likely a candidate. For now, we're considering learning the environment from scratch. If that proves to be difficult of a task, we can switch to the other extreme and then later verify a potential candidate.


![idea](https://user-images.githubusercontent.com/43754306/129492910-42a773d1-662e-47e8-ab2c-aa702a79a5d9.jpeg)


