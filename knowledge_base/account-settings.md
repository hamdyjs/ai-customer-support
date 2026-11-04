# Managing Your Account Settings

Your account settings are all under **Settings**, which you can open from your profile picture in the top-right corner of Flowdesk. This article covers each section in turn.

| Section       | What you can do there                                         |
| ------------- | ------------------------------------------------------------- |
| Profile       | Change your name, sign-in email, and time zone                |
| Security      | Change your password and turn on two-step verification        |
| Notifications | Choose which emails you receive                               |
| Team          | Invite and remove team members, and set roles on Business     |
| Integrations  | Connect and disconnect other services on Pro and Business     |
| Billing       | Manage your subscription, payment method, and invoices        |
| Account       | Update your company name, export data, and close your account |

## Profile

### Changing your name

Go to **Settings > Profile**, edit your name, and select **Save**. Your new name appears on your tasks and comments right away.

### Changing your sign-in email

1. Go to **Settings > Profile**.
2. Select **Change email**.
3. Enter your new address and your password.
4. Open the confirmation email we send to the new address and select the link within 24 hours.

Your old address keeps working until you confirm the new one. We also send a notice to the old address, so you'll know if someone else tries to change it.

Each email address can belong to only one Flowdesk account, so you can't use an address that's already registered. If you no longer have access to your old address, [open a support ticket](support-tickets.md). The support team will need to verify your identity before changing it.

### Time zone

Due dates and reminders use the time zone set in your profile. Billing dates and payment times are always in UTC.

## Security

### Changing your password

Go to **Settings > Security** and select **Change password**. Passwords must be at least 12 characters long. Changing your password signs you out on all your other devices.

If you've forgotten your password, see [Common Troubleshooting](troubleshooting.md#i-cant-sign-in).

### Two-step verification

Two-step verification protects your account with a code from an authenticator app as well as your password. We strongly recommend it, especially for account owners.

1. Go to **Settings > Security**.
2. Select **Turn on two-step verification**.
3. Scan the QR code with your authenticator app.
4. Enter the 6-digit code to confirm.
5. Save your recovery codes somewhere safe.

If you lose your device and your recovery codes, the support team will need to verify your identity before they can turn it off.

## Notifications

Choose what you're emailed about in **Settings > Notifications**. You can turn off task updates, mentions, and weekly summaries.

Security and billing emails can't be turned off.

## Team members

Prices are per account, so adding team members doesn't change your bill.

### Inviting someone

1. Go to **Settings > Team**.
2. Select **Invite member**.
3. Enter their email address, and on Business, choose a role.
4. Select **Send invite**.

Invitations expire after 7 days. You can resend one from the same page.

### Removing someone

Go to **Settings > Team**, select the person, and select **Remove from team**. Their tasks stay in your projects, and you can reassign them. Automation rules they created keep running, unless they depend on an integration they connected.

### Team permissions on Business

Team permissions are a Business feature. On Starter and Pro, only the owner can manage billing and the team, and everyone else has the same access as a Member.

| Role   | Projects         | Team settings | Billing |
| ------ | ---------------- | ------------- | ------- |
| Owner  | Full access      | Yes           | Yes     |
| Admin  | Full access      | Yes           | Yes     |
| Member | Create and edit  | No            | No      |
| Viewer | View and comment | No            | No      |

To change someone's role, go to **Settings > Team** and choose a new one next to their name. The change applies immediately. It's easiest to set the right role when you send the invite, as new members join as a Member by default.

### Transferring ownership

An account has one owner. To hand it over, the current owner goes to **Settings > Team**, selects **Transfer ownership**, and chooses another team member. The new owner must accept by email. The previous owner then becomes an Admin on Business, or a Member on other plans.

If the owner has left your company and can't do this, [open a support ticket](support-tickets.md). You'll need to show you're authorized to act for the company.

## Integrations

Connect other services, such as your calendar, in **Settings > Integrations**. If a connection keeps dropping, see [Common Troubleshooting](troubleshooting.md#integration-problems).

## Billing settings

The **Settings > Billing** section has four tabs.

- **Subscription:** your plan, billing cycle, and renewal date. See [Changing Your Flowdesk Subscription](subscription-changes.md).
- **Payment method:** your saved cards.
- **Invoices:** your invoices and payment attempts. See [Understanding Your Invoices](invoices.md).
- **Billing details:** your billing company name, address, and tax ID.

### Updating your card

Go to **Settings > Billing > Payment method** and select **Add card**. New cards become your main card, and you can remove the old one. If an invoice has failed, updating your card also triggers a new attempt, usually within a day.

Never send your card number in a support ticket or chat. The support team will never ask for it.

### Changing your billing email

Your billing email is where we send invoices and payment notices. It can be different from your sign-in email, and it doesn't need to be a Flowdesk user's address.

For security, the support team makes this change. [Open a support ticket](support-tickets.md) with the new address. Support will send a confirmation to the new address, then update it once that's confirmed. This usually takes up to 5 business days, and the ticket stays **In progress** until it's done. Invoices keep going to the old address until then.

## Account

### Company name

Change your company name in **Settings > Account**. To change what appears on invoices, update **Settings > Billing > Billing details** as well.

### Accounts that share a name

Names and company names don't have to be unique, so two different Flowdesk accounts can belong to people with the same name. Your account is identified by your sign-in email. Flowdesk support can only discuss the account you're signed in to, and we can't share details of any other account, even one with the same name.

### Account status

| Status    | What it means                                                                                                                                                      |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Active    | The account is in normal use. This includes canceled subscriptions with paid time remaining.                                                                       |
| Suspended | Access is restricted because an invoice is more than 30 days overdue. See [Failed Payments and Overdue Invoices](failed-payments.md#if-your-account-is-suspended). |
| Closed    | The account has been closed and can't be used.                                                                                                                     |

### Exporting your data

Go to **Settings > Account** and select **Export data**. Choose CSV or JSON, and we'll email you a download link when your export is ready. Large accounts can take a few hours. The link lasts for 7 days.

### Closing your account

Closing your account is permanent, and it isn't the same as canceling a subscription. It deletes your project data and removes access for your whole team straight away, rather than after the [90-day retention period](cancellation.md#what-happens-to-your-data).

Only the owner can request this:

1. Pay any open or overdue invoices.
2. Export anything you want to keep.
3. Go to **Settings > Account** and select **Close account**.
4. Confirm using the link we email you.

Closing your account doesn't refund any unused time. We keep billing records for as long as tax and accounting rules require.
