from argparse import ArgumentParser
import numpy as np
from aioch import Client
from asyncio import run
from catboost import CatBoostRegressor


frames = {
    'Day': np.timedelta64(24, 'h'),
    'Hour': np.timedelta64(1, 'h'),
    'FifteenMinutes': np.timedelta64(15, 'm'),
    'TenMinutes': np.timedelta64(10, 'm'),
    'FiveMinute': np.timedelta64(5, 'm'),
    'Minute': np.timedelta64(1, 'm')
}


def candles_query(pair, market='binance', frame='Hour', db='fx'):
    pair = f'{market}_{pair}'
    return f'''
        SELECT t.tm AS time, a.price AS open, t.lp AS low, t.hp AS high, b.price AS close
            FROM (
                SELECT toStartOf{frame}(time) AS tm, MIN(price) AS lp, MAX(price) AS hp, MIN(id) AS lid, MAX(id) AS hid
                    FROM {db}.{pair}_trades GROUP BY toStartOf{frame}(time)
            ) AS t
            INNER JOIN {db}.{pair}_trades AS a ON a.id = t.lid 
            INNER JOIN {db}.{pair}_trades AS b ON b.id = t.hid
            ORDER BY t.tm
    '''


def candles_to_dataset(candles, frame=np.timedelta64(1, 'h'), bars=3):
    times = (candles[0] - candles[0][0]) / frame
    size = len(times)
    prices = np.stack(candles[1:], 1)
    dataset = np.concatenate([prices[b:size - bars + b] for b in range(bars)], 1)
    times_ok = np.abs(times[1:] - times[:-1] - 1) < 1E-6
    times_ok = np.all([times_ok[b:size - bars + b] for b in range(bars - 1)], 0)
    dataset = dataset[times_ok]
    dataset = dataset - dataset[:, 4 * (bars - 2) + 3, np.newaxis]  # Normalize by penultimate close price
    return dataset[:, :4 * (bars - 1) - 1], dataset[:, -4:]


async def main():
    parser = ArgumentParser(description='Simple japanese candles prediction test')
    parser.add_argument('-d', type=str, default='localhost', required=False, help='Clickhouse database host')
    parser.add_argument('-f', type=str, default='FifteenMinutes', required=False, help='Time frame')
    parser.add_argument('-g', action='store_true', help='Fit with GPU')
    args = parser.parse_args()
    tt = 'GPU' if args.g else 'CPU'
    client = Client(args.d, settings={'use_numpy': True})
    frame = args.f
    candles = await client.execute(candles_query('btc_usdt', frame=frame), columnar=True)
    x, y = candles_to_dataset(candles, frames[frame])
    lo_model = CatBoostRegressor(iterations=5000, task_type=tt)
    lo_model.fit(x, y[:, 1])
    hi_model = CatBoostRegressor(iterations=5000, task_type=tt)
    hi_model.fit(x, y[:, 2])


if __name__ == '__main__':
    run(main())

