# Changing Your Flowdesk Subscription

You can move between plans or switch between monthly and annual billing at any time. This article explains what each plan includes, when a change takes effect, and how it affects your next charge.

Only the account owner can change a subscription. On the Business plan, Admins can also make changes.

## Plans and prices

Flowdesk has three plans. Prices are per account, not per user, and every price is in US dollars.

| Plan     | Description                          | Monthly price | Annual price |
| -------- | ------------------------------------ | ------------- | ------------ |
| Starter  | Essential project management         | $19.00        | $190.00      |
| Pro      | Advanced project management          | $49.00        | $490.00      |
| Business | Project management for growing teams | $99.00        | $990.00      |

### What each plan includes

**Starter**

- Task boards
- Basic reports
- Email support

**Pro** includes everything in Starter, plus:

- Advanced reports
- Integrations
- Automation

**Business** includes everything in Pro, plus:

- Team permissions
- Priority support

### Annual savings

An annual subscription costs the same as ten months of monthly billing. Over a year, that saves $38.00 on Starter, $98.00 on Pro, and $198.00 on Business.

## Before you change plans

Go to **Settings > Billing > Subscription** to see your current plan, billing cycle, and renewal date. Your renewal date is also the date your current billing period ends.

If you have an overdue invoice, pay it before changing your plan. You can't upgrade or switch billing cycles while an invoice is overdue. See [Failed Payments and Overdue Invoices](failed-payments.md).

## Upgrading your plan

An upgrade takes effect as soon as you confirm it, and you can use the new features straight away. To upgrade:

1. Go to **Settings > Billing > Subscription**.
2. Select **Change plan**.
3. Choose the new plan and review the amount shown.
4. Select **Confirm upgrade**.

### What you'll pay

The cost depends on when you upgrade.

**On your renewal date.** There's nothing to prorate. Your new plan starts with a new billing period, and you pay the new plan's full price. For example, a customer who moves from Starter to Pro on their October 1 renewal date pays $49.00 on October 1 instead of the usual $19.00. They pay $49.00 on each renewal after that.

**During a billing period.** You pay the price difference for the days left in the period, on a separate invoice. Your renewal date doesn't change, and you pay the new plan's full price from your next renewal.

The prorated amount is:

```text
(new plan price - current plan price) x (days remaining / days in the billing period)
```

### Proration examples

| Change                   | Billing period | Days remaining | Calculation            | Upgrade charge |
| ------------------------ | -------------- | -------------- | ---------------------- | -------------- |
| Starter to Pro, monthly  | 30 days        | 15             | ($49 - $19) x 15/30    | $15.00         |
| Pro to Business, monthly | 30 days        | 6              | ($99 - $49) x 6/30     | $10.00         |
| Pro to Business, annual  | 365 days       | 73             | ($990 - $490) x 73/365 | $100.00        |

### What happens to your work

Your task boards, reports, and other project data carry over when you upgrade. Nothing needs to be exported or set up again.

### How an upgrade looks in your billing history

Each plan has its own subscription record. When you upgrade, your old subscription shows as **Expired** and your new one shows as **Active**. This doesn't mean your service lapsed. Your invoice history stays complete, and older invoices remain linked to the old subscription.

## Downgrading your plan

A downgrade takes effect at the end of your current billing period. You keep your current plan and all its features until then, and your next charge is at the lower price. To downgrade:

1. Go to **Settings > Billing > Subscription**.
2. Select **Change plan**.
3. Choose the lower plan.
4. Select **Confirm downgrade**.

Downgrades don't generate a refund or credit for the current period.

### What you lose when you downgrade

Your data isn't deleted, but features that aren't part of the lower plan stop working on the day the downgrade takes effect.

| Moving to             | What changes                                                                                         |
| --------------------- | ---------------------------------------------------------------------------------------------------- |
| Pro from Business     | Custom team roles are removed and everyone except the owner becomes a Member. Priority support ends. |
| Starter from Pro      | Automation rules pause, integrations disconnect, and advanced reports become unavailable.            |
| Starter from Business | All of the above.                                                                                    |

If you upgrade again later, paused automation rules and saved advanced reports come back. You'll need to reconnect your integrations and reassign team roles.

### Changing your mind

You can undo a scheduled downgrade any time before it takes effect. Go to **Settings > Billing > Subscription** and select **Keep current plan**.

## Switching between monthly and annual billing

### From monthly to annual

The switch takes effect immediately and starts a new 12-month billing period. You pay the annual price, minus a credit for the unused days of your current month.

For example, say you're on Pro monthly with 15 days left in a 30-day period. The credit is $24.50, half of your $49.00 monthly price, so you pay $465.50 for the first year. Your renewal date moves to one year from the day you switched.

### From annual to monthly

The switch takes effect when your annual term ends. You keep annual billing until then, and your first monthly invoice is issued on the renewal date. There's no refund for the remainder of the annual term.

## Changing your plan and canceling

Canceling replaces any scheduled downgrade. To stop using Flowdesk rather than move to a cheaper plan, see [Canceling Your Subscription](cancellation.md).

## Frequently asked questions

### Why was I charged more than usual after upgrading?

Your first charge after an upgrade reflects the new plan's price, or a prorated amount if you upgraded during a billing period. You can see the amount and billing period on each invoice in **Settings > Billing > Invoices**.

### Can I be on two plans at once?

No. An account has one active subscription at a time.

### Can support change my plan for me?

The support team can help, but it's usually quicker to do it yourself. Flowdesk's chat assistant can tell you which plan you're on, but it can't change it.
