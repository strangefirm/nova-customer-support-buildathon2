# Retrieval evaluation

- Mode: **Keyword retrieval (BM25)**  ·  k = 3  ·  34 answerable questions, 6 unanswerable
- Labels are hand-written section-level judgements (`evals/rag_gold.yaml`); a small, author-labelled set, so treat the numbers as a regression gauge, not a benchmark.

| Metric | Value |
|---|---|
| Precision@3 | 34.8% |
| Recall@3 | 79.4% |
| Hit rate@3 (any relevant section retrieved) | 82.3% |
| MRR | 0.750 |
| nDCG@3 | 0.742 |
| Abstention on unanswerable questions | 3/6 (50%) |

## Questions that were not perfect

| Id | Question | Relevant | Retrieved |
|---|---|---|---|
| q01 | How long does standard delivery take? | shipping_policy#Delivery timelines | shipping_policy#Delivery timelines; shipping_policy#Late or delayed orders; product_faq#Out of stock items |
| q02 | Is shipping free on my order? | shipping_policy#Delivery timelines | shipping_policy#Tracking your order; shipping_policy#Late or delayed orders; shipping_policy#Order cancellation |
| q03 | How do I track my parcel? | shipping_policy#Tracking your order | shipping_policy#Late or delayed orders; shipping_policy#Tracking your order; shipping_policy#Missing items and wrong items |
| q04 | My package is a week late, what should I do? | shipping_policy#Late or delayed orders | refund_policy#Late order refunds; shipping_policy#Late or delayed orders; refund_policy#Return window |
| q05 | I typed the wrong delivery address, can I change it? | shipping_policy#Changing the delivery address | shipping_policy#Changing the delivery address; account_and_login#Updating your email address or name; shipping_policy#Missing items and wrong items |
| q06 | Can I cancel an order after placing it? | shipping_policy#Order cancellation | shipping_policy#Tracking your order; shipping_policy#Late or delayed orders; billing_and_payments#Promo codes |
| q07 | The box arrived with one of my items missing | shipping_policy#Missing items and wrong items | shipping_policy#Missing items and wrong items; shipping_policy#Late or delayed orders; refund_policy#Return window |
| q08 | Do you ship overseas and who pays the customs fees? | shipping_policy#Delivery timelines | shipping_policy#Delivery timelines; billing_and_payments#Accepted payment methods |
| q09 | How many days do I have to return something? | refund_policy#Return window | refund_policy#Refund timelines; refund_policy#Return window; refund_policy#How to return an item and get a refund |
| q10 | What are the steps to send an item back for a refund? | refund_policy#How to return an item and get a refund | product_faq#Out of stock items; shipping_policy#Missing items and wrong items; refund_policy#Damaged or defective items |
| q11 | When will the money reach my account after I return a produc | refund_policy#Refund timelines | refund_policy#Items that cannot be returned; product_faq#Product sizing; refund_policy#Damaged or defective items |
| q12 | The jacket I received is ripped, can I get a replacement? | refund_policy#Damaged or defective items | refund_policy#Refund timelines |
| q13 | Which things can I not return at all? | refund_policy#Items that cannot be returned | refund_policy#Late order refunds; refund_policy#Return window; refund_policy#How to return an item and get a refund |
| q14 | These shoes are the wrong size, can I swap them for another  | product_faq#Product sizing; refund_policy#Exchanges | product_faq#Product sizing |
| q15 | My parcel got lost in transit, do I get my money back? | refund_policy#Late order refunds; shipping_policy#Late or delayed orders | refund_policy#Late order refunds; product_faq#Out of stock items; shipping_policy#Late or delayed orders |
| q16 | I forgot my password and cannot log in | account_and_login#How to reset your password | account_and_login#How to reset your password; account_and_login#Locked account; account_and_login#Reset email did not arrive |
| q17 | The password reset email never showed up | account_and_login#Reset email did not arrive | account_and_login#How to reset your password; account_and_login#Reset email did not arrive; account_and_login#Locked account |
| q18 | My account is locked after too many wrong attempts | account_and_login#Locked account | account_and_login#Locked account; shipping_policy#Missing items and wrong items; account_and_login#Deleting your account |
| q19 | I lost the phone with my authenticator app | account_and_login#Two-factor authentication problems | account_and_login#Two-factor authentication problems; refund_policy#Late order refunds |
| q20 | How do I change the email address on my profile? | account_and_login#Updating your email address or name | account_and_login#Updating your email address or name; shipping_policy#Changing the delivery address; account_and_login#Reset email did not arrive |
| q21 | I want my account permanently deleted | account_and_login#Deleting your account | account_and_login#Deleting your account; account_and_login#Locked account; account_and_login#Updating your email address or name |
| q22 | Which credit cards do you accept? | billing_and_payments#Accepted payment methods | billing_and_payments#Accepted payment methods; refund_policy#Refund timelines; billing_and_payments#When your card is charged |
| q23 | At what point is my card actually charged? | billing_and_payments#When your card is charged | billing_and_payments#When your card is charged; billing_and_payments#Failed payments; billing_and_payments#Payment disputes and chargebacks |
| q24 | I can see two charges on my statement for one order | billing_and_payments#Duplicate or incorrect charges; billing_and_payments#When your card is charged | billing_and_payments#When your card is charged; billing_and_payments#Failed payments; billing_and_payments#Duplicate or incorrect charges |
| q25 | I want to file a chargeback with my bank | billing_and_payments#Payment disputes and chargebacks | billing_and_payments#Payment disputes and chargebacks; billing_and_payments#Failed payments; refund_policy#Refund timelines |
| q26 | My payment keeps failing at checkout | billing_and_payments#Failed payments | billing_and_payments#Failed payments; billing_and_payments#Accepted payment methods; account_and_login#Locked account |
| q27 | Can I use two promo codes on the same order? | billing_and_payments#Promo codes | billing_and_payments#Promo codes; account_and_login#Two-factor authentication problems; shipping_policy#Tracking your order |
| q28 | I need an invoice with my company tax number | billing_and_payments#Invoices | billing_and_payments#Invoices; billing_and_payments#Accepted payment methods; shipping_policy#Tracking your order |
| q29 | What does the warranty cover? | product_faq#Warranty | product_faq#Warranty; refund_policy#Items that cannot be returned; product_faq#Blenders and kitchen appliances |
| q30 | My blender jar is cracked and leaking | product_faq#Blenders and kitchen appliances; refund_policy#Damaged or defective items | product_faq#Blenders and kitchen appliances; product_faq#Product care |
| q31 | How can I get notified when a sold-out item is back? | product_faq#Out of stock items | product_faq#Out of stock items; refund_policy#Exchanges; shipping_policy#Missing items and wrong items |
| q33 | How long is the battery covered for? | product_faq#Batteries and chargers | product_faq#Warranty; product_faq#Batteries and chargers; product_faq#Blenders and kitchen appliances |
| q34 | What hours is your phone support open? | product_faq#Contacting support | product_faq#Contacting support; shipping_policy#Tracking your order; account_and_login#Reset email did not arrive |
