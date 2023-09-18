from utils.dataloader import dataloader

data, market_data = dataloader()
print(data.mean())
print(data.var())
print(data.corr())

print(market_data.mean())
print(market_data.var())
