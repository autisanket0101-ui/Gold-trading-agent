import MetaTrader5 as mt5

if not mt5.initialize():
    print("Connection failed:", mt5.last_error())
    raise SystemExit

account = mt5.account_info()
terminal = mt5.terminal_info()

print("Connected:", terminal.connected)
print("Trading allowed:", terminal.trade_allowed)
print("Server:", account.server)
print("Balance:", account.balance)

gold = [
    x.name for x in mt5.symbols_get()
    if "XAU" in x.name.upper() or "GOLD" in x.name.upper()
]
print("Gold symbols:", gold)

mt5.shutdown()