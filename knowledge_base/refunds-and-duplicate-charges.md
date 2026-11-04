# Refunds and Duplicate Charges

This article explains how to check whether you were really charged twice, how to report a duplicate charge, and when Flowdesk gives refunds.

Every refund is reviewed by the support team. Flowdesk's chat assistant can look through your payments with you, but it can't approve or promise a refund.

## Was I charged twice?

Go to **Settings > Billing > Invoices** and open the invoice for the charge you're asking about. Then compare what you see with this table.

| What the invoice shows                             | Is it a duplicate?                                |
| -------------------------------------------------- | ------------------------------------------------- |
| Two **successful** payments on the same invoice    | **Possibly yes.** Report it to support.           |
| One failed payment and one successful payment      | No. The failed attempt didn't charge you.         |
| Several failed payments and one successful payment | No. Only the successful payment charged you.      |
| Two invoices for different billing periods         | No. Those are two separate months or terms.       |
| A regular invoice and a smaller upgrade invoice    | No. The smaller one is a prorated upgrade charge. |

### A real duplicate charge

A duplicate charge usually looks like two successful payments of the same amount, made close together, on one invoice. For example, two $49.00 payments for one invoice, at 09:14 and 09:16 on November 1. You should only have paid once for that billing period, so the second payment needs review.

### Things that look like duplicates but aren't

**A retry after a failed payment.** If your first attempt failed and a later one succeeded, you'll see two attempts but only one charge. See [Failed Payments and Overdue Invoices](failed-payments.md).

**A pending authorization from your bank.** Your banking app may briefly show a pending entry beside the real charge. It normally disappears within a few business days.

**Charges close to the same date.** A payment at the end of one month and another at the start of the next can look like a double charge, particularly on a card statement. Check whether each one matches a different invoice and billing period.

**Two Flowdesk accounts.** If you have two accounts, each one bills separately. Check the invoices in both before reporting a duplicate.

## Reporting a duplicate charge

[Open a support ticket](support-tickets.md) with the category **Billing** and these details:

- The invoice number, such as `inv_123`
- The payment numbers of both successful payments, such as `pay_456` and `pay_457`
- The date and amount of each charge
- A screenshot of your bank statement, if the charges appear there, with any card or account numbers hidden

Please report duplicates within 90 days of the charge.

### What happens next

1. Support confirms both payments were successfully collected.
2. If they find a duplicate, support refunds the extra payment in full.
3. We email you to confirm, and you can reply to the ticket with any questions.

We aim to finish the review within 2 business days. Tickets about an incorrect charge are treated as high priority.

## Refund policy

### Summary

| Situation                                  | Refund?                                           |
| ------------------------------------------ | ------------------------------------------------- |
| Confirmed duplicate payment                | Full refund of the extra payment                  |
| Annual plan, within 30 days of the payment | Full refund if you cancel, after review           |
| Annual plan, after 30 days                 | No refund. Access continues until the term ends   |
| Monthly plan renewal                       | No refund. Access continues until the period ends |
| Unused time after canceling                | No refund                                         |
| Unused time after a downgrade              | No refund or credit                               |
| Billing error by Flowdesk                  | The incorrect amount is refunded, after review    |

### Annual plans

You can request a full refund within 30 days of your annual payment, for a first year or a renewal. To do this, cancel your subscription and [open a support ticket](support-tickets.md). If the refund is approved, your access ends when it's issued rather than at the end of the term.

After 30 days, annual payments aren't refundable, but you keep access for the rest of the term.

### Monthly plans

We don't refund monthly payments or part-used months, so if you cancel partway through a month, you keep access until the period ends. See [Canceling Your Subscription](cancellation.md).

### Other requests

If something unusual happened, such as a long outage or a charge you can't explain, it's still worth asking. The support team reviews these requests case by case, though a refund isn't guaranteed.

## How refunds are paid

Refunds always go back to the original payment method. We can't send them to a different card or bank account.

Once a refund is issued, it usually takes **5 to 10 business days** to appear, depending on your bank. If your card has since been canceled or replaced, your bank will normally pass the money on to your new card or account. Ask your bank if you don't see it after 10 business days.

### How a refund appears in Flowdesk

We send a confirmation email when a refund is issued, with the amount and the payment it relates to. The original payment still shows as **Succeeded** on the invoice, and the invoice still shows as **Paid**, because those records describe the original charge. A refund doesn't change your subscription unless you've canceled.

## Disputing a charge with your bank

Please contact us before filing a dispute or chargeback. We can usually sort it out faster, and a chargeback can pause your account while your bank investigates.
