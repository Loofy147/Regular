import time
from decimal import Decimal
from genieune_heads.core import decimal_sin_cos

start = time.time()
for _ in range(1000):
    decimal_sin_cos(Decimal('1234567.89'))
end = time.time()
print(f"1000 calls took: {end - start:.4f}s")
print(f"Estimated 2M calls: {(end - start) * 2000:.2f}s")
