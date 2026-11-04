# Failed Payments and Overdue Invoices

If we can't take a payment, your service isn't interrupted right away. You'll have time to fix the problem, and we'll email you along the way.

**A failed payment doesn't charge you.** No money leaves your account when an attempt fails.

## Quick fix

1. Go to **Settings > Billing > Payment method** and check your card is up to date.
2. Update the card if it's expired or if you want to use a different one.
3. Go to **Settings > Billing > Invoices** and check whether the invoice is now paid. If you've just updated your card, allow up to a day for the new attempt.

To pay right away, open the invoice and select **Pay now**.

## Why payments fail

The reason is shown next to the failed attempt on your invoice.

| Reason shown         | What it means                                         | What to do                                                  |
| -------------------- | ----------------------------------------------------- | ----------------------------------------------------------- |
| `expired_card`       | The card on file has passed its expiry date           | Add the new card details or a different card                |
| `insufficient_funds` | The bank declined the charge because of the balance   | Add funds, or use a different card, then select **Pay now** |
| `card_declined`      | The bank declined the charge without giving a reason  | Contact your bank, or use a different card                  |
| `processing_error`   | A temporary problem occurred with the payment network | Wait for the automatic retry, or select **Pay now**         |

### Why does my bank show a charge for a failed payment?

Some banks briefly show a pending authorization when an attempt fails. It isn't a completed charge, and it normally disappears within a few business days. Your invoice shows what actually happened. Only attempts marked **Succeeded** took money from you.

### Why isn't my declined attempt listed?

If your card is rejected while you're entering it on the payment form, it never becomes a payment attempt, so it won't show on the invoice. Try another card, or ask your bank why it was declined.

## What happens after a payment fails

Here's the timeline for an invoice issued on day 0.

| Day       | What happens                                                                            | Invoice status |
| --------- | --------------------------------------------------------------------------------------- | -------------- |
| 0         | The invoice is issued and we make the first payment attempt. If it fails, we email you. | Open           |
| 3         | We retry automatically, once. If this fails, we email you again.                        | Open           |
| 7         | The invoice is due.                                                                     | Open           |
| 8         | The invoice becomes overdue and we send the first overdue notice.                       | Overdue        |
| 17 and 32 | We send further overdue notices.                                                        | Overdue        |
| 37        | If it's still unpaid, your account is suspended.                                        | Overdue        |

After the day 3 retry, we don't try again automatically unless you update your card. It's up to you to pay the invoice or add a working card.

### Example: an expired card

Your card expires, and the October 1 attempt fails with `expired_card`. You get an email and update your card the next day. We try again, and the payment succeeds on October 3. The invoice shows one failed and one successful attempt, and it's marked **Paid**. You were charged once.

### Example: insufficient funds

The attempts on October 1 and October 4 both fail with `insufficient_funds`. The invoice is due on October 8, and it becomes **Overdue** once that date has passed. Your account remains active, but you'll need to pay the invoice to avoid suspension.

## Overdue invoices

An invoice becomes overdue when it's unpaid after its due date, 7 days after it was issued.

### Will my account be affected?

Not straight away. You and your team keep full access for 30 days after the due date. If the invoice is still unpaid at that point, your account is suspended.

### New invoices while one is overdue

We keep issuing an invoice for each new billing period, because your subscription continues. Automatic payment pauses while you have an overdue invoice, so new invoices stay **Open** until you pay them, and they can become overdue too.

When you've paid the overdue invoice, automatic payment resumes for future invoices. Any invoices that are still open need to be paid with **Pay now**.

### Paying an overdue invoice

1. Go to **Settings > Billing > Invoices**.
2. Select the overdue invoice.
3. Select **Pay now**.
4. Confirm your saved card or enter a new one.

The invoice changes to **Paid** as soon as the payment succeeds. Pay the oldest invoice first.

### Need more time?

[Open a support ticket](support-tickets.md) before the suspension date. The support team can review your situation, but payment extensions aren't guaranteed.

## If your account is suspended

While your account is suspended:

- You can sign in, but you can only reach the Billing section of Settings.
- Your team can't view or edit projects.
- Automation rules stop running, and integrations stop syncing.
- Your data is kept safe, and nothing is deleted.

### Getting your account back

Pay **all** your overdue invoices. Your account normally becomes active again within an hour of the last payment succeeding. Automation rules and integrations resume by themselves, though you may need to reconnect some integrations.

### If you take no action

If your account stays suspended for 90 days, your subscription is canceled. The 90-day data retention period described in [Canceling Your Subscription](cancellation.md#what-happens-to-your-data) then begins, and your data is deleted when it ends.

## Preventing failed payments

- Update your card before it expires.
- Add a backup card in **Settings > Billing > Payment method**. We'll only use it if your main card fails.
- Tell your bank to expect a charge if you've moved to an annual plan, as large one-off charges are more likely to be blocked.
- Make sure your billing email goes to someone who reads it, so failure notices aren't missed.

## Think you were charged twice?

A failed attempt followed by a successful one isn't a double charge. If your invoice shows **two successful payments**, see [Refunds and Duplicate Charges](refunds-and-duplicate-charges.md).
