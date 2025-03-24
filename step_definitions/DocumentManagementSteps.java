package step_definitions;

import io.cucumber.java.en.*;
import org.openqa.selenium.*;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.Select;
import org.junit.Assert;
import java.time.LocalDate;

public class DocumentManagementSteps {
    private WebDriver driver;
    private WebDriverWait wait;

    @Given("^user logs in with username \"([^\"]*)\" and password \"([^\"]*)\"$")
    public void userLogsInWithCredentials(String username, String password) {
        driver.get("https://example.com/login");
        WebElement usernameField = driver.findElement(By.id("username"));
        WebElement passwordField = driver.findElement(By.id("password"));
        WebElement loginButton = driver.findElement(By.id("login-btn"));
        
        usernameField.sendKeys(username);
        passwordField.sendKeys(password);
        loginButton.click();
        
        wait.until(ExpectedConditions.urlContains("/dashboard"));
    }

    @When("^user navigates to case \"([^\"]*)\"$")
    public void userNavigatesToCase(String caseNumber) {
        WebElement searchBox = driver.findElement(By.id("case-search"));
        searchBox.sendKeys(caseNumber);
        searchBox.sendKeys(Keys.ENTER);
        wait.until(ExpectedConditions.elementToBeClickable(By.linkText(caseNumber))).click();
    }

    @When("^user clicks on \"([^\"]*)\" button$")
    public void userClicksButton(String buttonText) {
        WebElement button = wait.until(ExpectedConditions.elementToBeClickable(
            By.xpath("//button[contains(text(),'" + buttonText + "')]")));
        button.click();
    }

    @When("^user uploads file \"([^\"]*)\" to the system$")
    public void userUploadsFile(String fileName) {
        WebElement fileInput = driver.findElement(By.cssSelector("input[type='file']"));
        String filePath = System.getProperty("user.dir") + "/test-files/" + fileName;
        fileInput.sendKeys(filePath);
        wait.until(ExpectedConditions.presenceOfElementLocated(By.className("upload-progress")));
    }

    @When("^user enters document title \"([^\"]*)\"$")
    public void userEntersDocumentTitle(String title) {
        WebElement titleField = driver.findElement(By.id("doc-title"));
        titleField.clear();
        titleField.sendKeys(title);
    }

    @When("^user selects document type \"([^\"]*)\" from dropdown$")
    public void userSelectsDocumentType(String docType) {
        Select dropdown = new Select(driver.findElement(By.id("doc-type")));
        dropdown.selectByVisibleText(docType);
    }

    @When("^user adds tags \"([^\"]*)\" to the document$")
    public void userAddsTags(String tags) {
        WebElement tagField = driver.findElement(By.id("tag-input"));
        for (String tag : tags.split(",")) {
            tagField.sendKeys(tag.trim());
            tagField.sendKeys(Keys.ENTER);
        }
    }

    @When("^user enters case number \"([^\"]*)\"$")
    public void userEntersCaseNumber(String caseNumber) {
        WebElement caseField = driver.findElement(By.id("case-number"));
        caseField.sendKeys(caseNumber);
    }

    @When("^user selects case type \"([^\"]*)\"$")
    public void userSelectsCaseType(String caseType) {
        Select typeDropdown = new Select(driver.findElement(By.id("case-type")));
        typeDropdown.selectByVisibleText(caseType);
    }

    @When("^user enters case title \"([^\"]*)\"$")
    public void userEntersCaseTitle(String title) {
        WebElement titleField = driver.findElement(By.id("case-title"));
        titleField.sendKeys(title);
    }

    @When("^user selects jurisdiction \"([^\"]*)\"$")
    public void userSelectsJurisdiction(String jurisdiction) {
        Select jurisdictionDropdown = new Select(driver.findElement(By.id("jurisdiction")));
        jurisdictionDropdown.selectByVisibleText(jurisdiction);
    }

    @When("^user enters filing date \"([^\"]*)\"$")
    public void userEntersFilingDate(String date) {
        WebElement dateField = driver.findElement(By.id("filing-date"));
        dateField.sendKeys(date);
    }

    @When("^user adds team member \"([^\"]*)\" as \"([^\"]*)\"$")
    public void userAddsTeamMember(String email, String role) {
        WebElement emailField = driver.findElement(By.id("team-member-email"));
        Select roleDropdown = new Select(driver.findElement(By.id("team-member-role")));
        WebElement addButton = driver.findElement(By.id("add-team-member"));
        
        emailField.sendKeys(email);
        roleDropdown.selectByVisibleText(role);
        addButton.click();
    }

    @When("^user adds comment \"([^\"]*)\" at page \"([^\"]*)\"$")
    public void userAddsComment(String comment, String page) {
        WebElement commentField = driver.findElement(By.id("comment-input"));
        WebElement pageField = driver.findElement(By.id("page-number"));
        
        pageField.sendKeys(page);
        commentField.sendKeys(comment);
        driver.findElement(By.id("add-comment")).click();
    }

    @When("^user highlights text \"([^\"]*)\" in yellow$")
    public void userHighlightsText(String text) {
        String script = "window.getSelection().selectAllChildren(document.querySelector('p:contains(\"" + text + "\")'))";
        ((JavascriptExecutor) driver).executeScript(script);
        driver.findElement(By.id("highlight-yellow")).click();
    }

    @Then("^system displays message \"([^\"]*)\"$")
    public void systemDisplaysMessage(String message) {
        WebElement alert = wait.until(ExpectedConditions.presenceOfElementLocated(
            By.className("alert-message")));
        Assert.assertEquals(message, alert.getText());
    }

    @Then("^document \"([^\"]*)\" appears in case files list$")
    public void documentAppearsInList(String docTitle) {
        WebElement filesList = driver.findElement(By.id("case-files-list"));
        Assert.assertTrue(filesList.getText().contains(docTitle));
    }

    @Then("^document status shows \"([^\"]*)\"$")
    public void documentStatusShows(String status) {
        WebElement statusElement = driver.findElement(By.className("doc-status"));
        Assert.assertEquals(status, statusElement.getText());
    }

    @Then("^system shows \"([^\"]*)\" team members for the case$")
    public void systemShowsTeamMembers(String count) {
        WebElement teamList = driver.findElement(By.id("team-members-list"));
        Assert.assertEquals(Integer.parseInt(count), 
            teamList.findElements(By.tagName("li")).size());
    }

    @Then("^notification is sent to added team members$")
    public void notificationIsSentToTeamMembers() {
        WebElement notificationCenter = driver.findElement(By.id("notification-center"));
        Assert.assertTrue(notificationCenter.getText().contains("New team member added"));
    }
} 