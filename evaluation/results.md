# Routing evaluation results

- Model: `openai/gpt-oss-120b`
- Cases: 26
- Correct: 24
- Accuracy: **92.3%**
- Run at: 2026-11-18 22:50 UTC

| # | Message | Expected | Actual | |
|---|---|---|---|---|
| 1 | What albums do you have by the Rolling Stones? | `music_catalog` | `music_catalog` | pass |
| 2 | List some songs by AC/DC. | `music_catalog` | `music_catalog` | pass |
| 3 | Do you have anything by Miles Davis? | `music_catalog` | `music_catalog` | pass |
| 4 | What jazz songs are in the catalog? | `music_catalog` | `music_catalog` | pass |
| 5 | Is there a song called 'Black Dog'? | `music_catalog` | `music_catalog` | pass |
| 6 | Recommend something similar to Led Zeppelin. | `music_catalog` | `music_catalog` | pass |
| 7 | Which genres do you sell? | `music_catalog` | `music_catalog` | pass |
| 8 | Show me every album by Iron Maiden. | `music_catalog` | `music_catalog` | pass |
| 9 | How much was my most recent purchase? | `invoice_info` | `invoice_info` | pass |
| 10 | What did I buy last month? | `invoice_info` | `invoice_info` | pass |
| 11 | List all my invoices. | `invoice_info` | `invoice_info` | pass |
| 12 | What was the most expensive track I ever bought? | `invoice_info` | `both` | **miss** |
| 13 | Who was the support rep on invoice 382? | `invoice_info` | `invoice_info` | pass |
| 14 | What is the total of my biggest invoice? | `invoice_info` | `invoice_info` | pass |
| 15 | Who is my support representative? | `invoice_info` | `invoice_info` | pass |
| 16 | What did I pay for my last order, and who handled it? | `invoice_info` | `invoice_info` | pass |
| 17 | How much was my most recent purchase, and what albums do you have by the Rolling Stones? | `both` | `both` | pass |
| 18 | Of the albums I have bought, which ones are by AC/DC? | `both` | `both` | pass |
| 19 | Show my invoice history and recommend a jazz album. | `both` | `both` | pass |
| 20 | Did I buy the album 'No Security', and what did it cost? | `both` | `invoice_info` | **miss** |
| 21 | Which tracks did I buy from the Rolling Stones, and do you have their other albums? | `both` | `both` | pass |
| 22 | What genres do my purchased tracks belong to? | `both` | `both` | pass |
| 23 | Thanks, that's all for now. | `FINISH` | `FINISH` | pass |
| 24 | Perfect, no more questions. | `FINISH` | `FINISH` | pass |
| 25 | That answers it - bye. | `FINISH` | `FINISH` | pass |
| 26 | Great, that's everything I needed. | `FINISH` | `FINISH` | pass |

## Misrouted cases

- `invoice_info` → `both`: What was the most expensive track I ever bought?  
  purchase line items
- `both` → `invoice_info`: Did I buy the album 'No Security', and what did it cost?  
  catalog existence plus purchase price
