import pandas as pd
import talib

def signal(dataframe):
    aroondown, _ = talib.AROON(
        dataframe["high"].to_numpy(dtype=float),
        dataframe["low"].to_numpy(dtype=float),
        timeperiod=14,
    )
    down = pd.Series(aroondown, index=dataframe.index)
    return down.gt(70.0).fillna(False).astype(bool)
