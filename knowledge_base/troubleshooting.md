# Common Troubleshooting

Start here if something in Flowdesk isn't working as expected. For most problems, the basic checks below are enough.

## Try these first

1. Check the status page at status.flowdesk.example for any ongoing incidents.
2. Refresh the page, or close and reopen the mobile app.
3. Make sure you're using the latest version of Chrome, Firefox, Safari, or Edge, or the latest mobile app.
4. Try a private browser window. If the problem disappears, a browser extension or stored data is probably causing it.
5. Check that your plan includes the feature, and that your account is active.

## Is the feature on your plan?

A missing feature often isn't a bug. Features can disappear after a downgrade, or when an account is suspended.

| Feature          | Starter | Pro | Business |
| ---------------- | ------- | --- | -------- |
| Task boards      | Yes     | Yes | Yes      |
| Basic reports    | Yes     | Yes | Yes      |
| Advanced reports | No      | Yes | Yes      |
| Integrations     | No      | Yes | Yes      |
| Automation       | No      | Yes | Yes      |
| Team permissions | No      | No  | Yes      |
| Email support    | Yes     | Yes | Yes      |
| Priority support | No      | No  | Yes      |

To compare plans or upgrade, see [Changing Your Flowdesk Subscription](subscription-changes.md).

## Sign-in problems

### I can't sign in

Work through these in order.

1. **Check your email address.** Use the address you sign in with, which may not be your billing email.
2. **Reset your password.** Select **Forgot password** on the sign-in page. The link lasts for 60 minutes. Check your spam folder if it doesn't arrive within a few minutes.
3. **Wait if you're locked out.** After 5 incorrect passwords, sign-in is blocked for 15 minutes.
4. **Check your invitation.** If you're new to the team, your invitation may have expired after 7 days. Ask your owner or an Admin to resend it.

If none of this helps, email support@flowdesk.example from the address on your account.

### I lost my two-step verification device

Use one of your recovery codes on the sign-in page. If you don't have them, [open a support ticket](support-tickets.md), or email us if you can't sign in. Support will need to verify your identity.

### I can only see the Billing page

Your account is suspended because of an overdue invoice. See [Failed Payments and Overdue Invoices](failed-payments.md#if-your-account-is-suspended). Paying it normally restores access within an hour.

### I can't edit anything

Your projects are read-only if your subscription has ended. See [Canceling Your Subscription](cancellation.md#what-happens-to-your-data). On Business, you might also have the Viewer role, which your owner or an Admin can change.

## Automation rules aren't running

Automation is available on Pro and Business. If a rule has stopped running, these are the usual causes, most likely first.

### 1. Check the rule's run history

Go to **Automation**, select the rule, and open **Run history**. It tells you which of these situations you're in.

- **No recent runs:** the rule isn't being triggered. Check its status and trigger.
- **Failed runs:** the rule is being triggered but can't finish. The error tells you why.
- **Successful runs:** the rule is working. Check whether its conditions are narrower than you intended.

### 2. Check that the rule is on

Rules can be paused by mistake, and they're paused automatically after a downgrade to Starter. If the rule shows as **Paused**, turn it back on.

### 3. Check the rule's trigger and conditions

Rules run only when the trigger happens, and only if every condition is true. If someone has renamed a project, moved a board column, or deleted a label the rule depends on, it may no longer match. Edit the rule and choose the current value.

### 4. Check any connected integrations

If the rule uses an integration that has disconnected, each run fails. Go to **Settings > Integrations** and reconnect it. See [An integration keeps asking me to reconnect](#an-integration-keeps-asking-me-to-reconnect).

### 5. Check for a removed team member

If a rule relies on an integration connected by someone who's since left the team, reconnect it using another person's account.

### 6. Check your billing

Automation rules stop running while an account is suspended. They restart by themselves when it's active again.

### Still not running?

[Open a support ticket](support-tickets.md) with the rule's name, when it last ran, and any error from its run history. If stopped rules are blocking your work, say so and we'll treat your ticket as high priority.

## Integration problems

### An integration keeps asking me to reconnect

This is most common with calendar integrations. It means Flowdesk's permission to access the other service keeps being lost or rejected. Try these steps.

1. Go to **Settings > Integrations**, select the integration, and select **Disconnect**.
2. Sign out of the other service, such as your calendar provider, and sign in with the account you want to use.
3. Go back to Flowdesk and select **Connect**.
4. Approve **all** the requested permissions. A partial approval can bring this prompt back.

If the prompt keeps coming back, check these as well.

- **Changed password:** changing your password with the other service can cancel its connections.
- **Company policy:** your IT admin may limit which apps can connect, or how long connections last. Ask them to approve Flowdesk.
- **Multiple accounts:** you may have connected with a different account from the one that holds your calendar.
- **Blocked cookies:** browser settings or extensions that block third-party cookies can stop the connection finishing. Try a private window with extensions turned off.

### An integration disappeared

Integrations disconnect when an account moves to Starter. After upgrading back to Pro or Business, you'll need to reconnect them.

### Data isn't syncing

Syncing can take up to 15 minutes. If it's taking longer, check that the integration shows as **Connected**, then select **Sync now**.

## Report problems

### Reports crash or won't load

Very large reports are the most common cause.

1. Shorten the date range. Try a month rather than a year.
2. Filter the report to fewer projects or team members.
3. Clear your browser's cache for Flowdesk, or update the mobile app.
4. Try a different browser or device.

If a report crashes the app every time, even with a short date range, [open a support ticket](support-tickets.md). Tell us the report's name, the device and browser or app version you're using, and what you see when it fails.

### My reports are missing

Reports carry over when you upgrade, so an upgrade won't remove them. After a downgrade to Starter, advanced reports aren't available, but they come back if you upgrade again. Basic reports stay available on all plans.

### A report's numbers look wrong

Check the report's date range, filters, and time zone. Reports use the time zone in your profile, so tasks completed near midnight can fall on a different day from the one you expect.

## Email problems

### I'm not receiving Flowdesk emails

Check your spam folder, then add flowdesk.example to your safe senders. Check **Settings > Notifications** for task emails. Company email filters can also block automated messages, so your IT admin may need to allow them.

### Invoices go to the wrong person

Invoices go to your billing email. See [Changing your billing email](account-settings.md#changing-your-billing-email).

### I got a payment failure email

Your card needs attention. See [Failed Payments and Overdue Invoices](failed-payments.md#quick-fix).

## Performance

If Flowdesk feels slow, close unused tabs, turn off browser extensions, and check your network connection. Boards with thousands of tasks load more quickly if you archive completed work.

## Mobile app

Most mobile problems are fixed by updating the app, or by signing out and back in. Billing and team settings are only available on the web.

## Still need help?

[Open a support ticket](support-tickets.md) with what you tried, and we'll take it from there. Please include any error messages and screenshots.
