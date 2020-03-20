### Schema Networks: Zero-shot Transfer with a Generative Causal Model of Intuitive Physics

An object oriented generative model capable of disentangling multiple causes of events and reasoning backward through causes to achieve goals.

### How to try it

Install required packages by running `pip install -r requirements.txt`

Tweak some options in `model/constants.py`, such as:

- `USE_HANDCRAFTED_ATTRIBUTE_SCHEMAS` / `USE_HANDCRAFTED_REWARD_SCHEMAS` - use handcrafted vectors instead of learned
- `VISUALIZE_*` - visualize stuff

Run `python3 run_agent.py`


### References
- [Blog post: General Game Playing with Schema Networks](https://www.vicarious.com/general-game-playing-with-schema-networks.html)
- [Paper: Kansky, Silver, Mély, Eldawy, Lázaro-Gredilla, Lou, Dorfman, Sidor, Phoenix and George. 2017.](https://www.vicarious.com/img/icml2017-schemas.pdf)
