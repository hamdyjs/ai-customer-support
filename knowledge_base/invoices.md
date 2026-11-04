# Understanding Your Invoices

Flowdesk issues an invoice at the start of every billing period. This article explains how to find your invoices, what they contain, and how to read their statuses.

## Finding and downloading invoices

All of your invoices are in **Settings > Billing > Invoices**, newest first.

1. Find the invoice by its date or invoice number.
2. Select the invoice to see its details and payment attempts.
3. Select **Download PDF**.

To download several invoices for your accountant, tick the ones you need and select **Download selected**. You'll get a single ZIP file.

The account owner can always see invoices. On the Business plan, Admins can see them too.

## When invoices are issued

We bill in advance, so each invoice pays for the period ahead.

- **Monthly plans** get one invoice a month, issued on the first day of the billing period.
- **Annual plans** get one invoice a year, issued on the first day of the annual term. You won't see any invoices in the months between.

Every invoice is due **7 days after it's issued**. We try to take payment automatically on the day it's issued, so most invoices are paid long before they're due.

A copy of each invoice is emailed to your billing email address.

## What's on an invoice

Each invoice shows:

- An invoice number, such as `inv_123`
- The plan and billing cycle
- The amount, in US dollars
- The issue date and due date
- The billing period it covers
- Your company name and billing details
- Your tax ID, if you've added one
- The status

### Understanding the billing period

The billing period includes its start date and runs up to its end date. An invoice with a period of October 1 to November 1 covers all of October, and the next period starts on November 1.

## Invoice statuses

| Status  | What it means                                           | What you need to do                                       |
| ------- | ------------------------------------------------------- | --------------------------------------------------------- |
| Open    | Issued and not paid yet, and the due date hasn't passed | Nothing, unless automatic payment has failed or is paused |
| Paid    | At least one payment has succeeded                      | Nothing                                                   |
| Overdue | Not paid, and the due date has passed                   | Pay it as soon as you can                                 |

If an invoice is overdue, see [Failed Payments and Overdue Invoices](failed-payments.md).

## Payment attempts on an invoice

Open an invoice to see every attempt to collect payment, each with its own payment number, such as `pay_456`, its date and time in UTC, and its result.

| Result    | What it means                                          |
| --------- | ------------------------------------------------------ |
| Succeeded | The money was taken from your payment method           |
| Failed    | The payment didn't go through, and you weren't charged |

A failed attempt still lists the invoice amount, because that's the amount we tried to collect. It doesn't mean the money left your account.

More than one attempt on an invoice is usually normal. For example, a failed attempt on October 1 followed by a successful one on October 3 means you paid once. Two **successful** attempts on one invoice is different, and could be a duplicate charge. See [Refunds and Duplicate Charges](refunds-and-duplicate-charges.md).

## Why an invoice amount changed

If an invoice is higher or lower than you expected, one of these is the most likely reason.

| What you see                               | Likely reason                                                                 |
| ------------------------------------------ | ----------------------------------------------------------------------------- |
| A higher amount from one month to the next | You upgraded. Starter to Pro raises the monthly charge from $19.00 to $49.00. |
| A lower amount from one month to the next  | A downgrade took effect at your renewal date.                                 |
| An extra invoice for a small amount        | You upgraded during a billing period and paid a prorated difference.          |
| A much larger charge once a year           | You're on an annual plan, such as Pro at $490.00.                             |
| An annual charge below the listed price    | You switched from monthly to annual, and unused days were credited.           |

For the full details, see [Changing Your Flowdesk Subscription](subscription-changes.md).

## Billing details on your invoices

### Adding a VAT number or tax ID

Add your tax ID in **Settings > Billing > Billing details**. It appears on all future invoices. You can also ask support to add it for you.

Past invoices aren't updated automatically. If you need an earlier invoice reissued with your tax ID, [open a support ticket](support-tickets.md) and include the invoice number.

### Taxes

Flowdesk doesn't currently add VAT or sales tax to invoices, so the amount you pay matches the plan price. Your tax ID is shown for your own records. Please check with your accountant how to treat these invoices.

### Changing your company name or address

Update these in **Settings > Billing > Billing details**. The changes apply to future invoices.

### Sending invoices to someone else

Invoices go to your billing email address, which can be different from your sign-in email. You could use a shared address for your finance team, for example. See [Changing your billing email](account-settings.md#changing-your-billing-email).

## Currency

All invoices are in US dollars. Your bank may show a different amount if it converts the charge to your local currency, and it may add its own fee.

## Paying an invoice manually

Automatic payment handles most invoices. If an invoice is open or overdue, you can pay it yourself:

1. Go to **Settings > Billing > Invoices**.
2. Select the invoice.
3. Select **Pay now**.
4. Confirm your saved card or enter a new one.

We accept Visa, Mastercard, and American Express credit and debit cards.

## Can't find an invoice?

Check that you're signed in to the right account, as invoices belong to one account only. If you have more than one Flowdesk account, check each of them. If it's still missing, [open a support ticket](support-tickets.md) with the billing period and amount.
