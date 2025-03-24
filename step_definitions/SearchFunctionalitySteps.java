package stepdefinitions;

import io.cucumber.java.en.*;
import org.junit.Assert;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;

public class SearchFunctionalitySteps {
    
Here is a complete Java step definition implementation for the Cucumber step "Given the search bar is visible":

```java
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import io.cucumber.java.en.Given;

public class SearchBarSteps {
    private WebDriver driver;
    private WebDriverWait wait;

    public SearchBarSteps(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, 10); // Adjust timeout as necessary
    }

    @Given("^the search bar is visible$")
    public void theSearchBarIsVisible() {
        WebElement searchBar = driver.findElement(By.id("search-bar")); // Adjust the locator as necessary
        wait.until(ExpectedConditions.visibilityOf(searchBar));
    }
}
```

### Explanation:
- The `@Given` annotation is used to define the step that checks if the search bar is visible.
- The `theSearchBarIsVisible` method locates the search bar element using its ID (you may need to adjust the locator based on your application's HTML).
- The `WebDriverWait` is used to wait until the search bar is visible on the page, ensuring that the test does not proceed until the element is ready for interaction.
Here is the complete Java step definition implementation for the Cucumber step "When the user enters 'laptop' in the search bar":

```java
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import io.cucumber.java.en.When;

public class SearchSteps {
    private WebDriver driver;
    private WebDriverWait wait;

    public SearchSteps(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, 10); // Adjust timeout as needed
    }

    @When("^the user enters \"([^\"]*)\" in the search bar$")
    public void theUserEntersSearchTermInTheSearchBar(String searchTerm) {
        WebElement searchBar = driver.findElement(By.id("search-bar")); // Adjust the locator as needed
        wait.until(ExpectedConditions.elementToBeClickable(searchBar));
        searchBar.clear();
        searchBar.sendKeys(searchTerm);
    }
}
```

### Explanation:
- The `@When` annotation specifies the Cucumber step with the regex pattern to capture the search term.
- The method `theUserEntersSearchTermInTheSearchBar` takes a parameter `searchTerm`, which corresponds to the quoted string in the step.
- The `WebElement` for the search bar is located using its ID (you may need to adjust the locator based on your application's HTML).
- The `WebDriverWait` is used to ensure the search bar is clickable before interacting with it.
- The search bar is cleared and the search term is entered using `sendKeys()`.
public void userClicksButton(String buttonText) {
WebElement button = wait.until(ExpectedConditions.elementToBeClickable(
            By.xpath("//button[contains(text(),'" + buttonText + "')]")));
        button.click();
}
Here is the complete Java step definition implementation for the Cucumber step "Then the search results should display items related to 'laptop'":

```java
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import io.cucumber.java.en.Then;

public class SearchResultsStepDefinitions {
    private WebDriver driver;
    private WebDriverWait wait;

    public SearchResultsStepDefinitions(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, 10); // Adjust timeout as needed
    }

    @Then("^the search results should display items related to \"([^\"]*)\"$")
    public void theSearchResultsShouldDisplayItemsRelatedTo(String searchTerm) {
        // Wait for the search results container to be visible
        WebElement resultsContainer = wait.until(ExpectedConditions.visibilityOfElementLocated(By.id("search-results-container")));

        // Check if the results contain the search term
        boolean isRelatedItemDisplayed = resultsContainer.getText().toLowerCase().contains(searchTerm.toLowerCase());

        // Assert that the search term is present in the results
        if (!isRelatedItemDisplayed) {
            throw new AssertionError("Expected search results to contain items related to: " + searchTerm);
        }
    }
}
```

### Explanation:
1. **Imports**: Necessary Selenium and Cucumber imports are included.
2. **WebDriver and WebDriverWait**: The constructor initializes the WebDriver and WebDriverWait for handling dynamic waits.
3. **Step Definition**: The method `theSearchResultsShouldDisplayItemsRelatedTo` captures the search term using the specified regex pattern.
4. **Visibility Wait**: It waits for the search results container to be visible before proceeding.
5. **Text Check**: It checks if the search term is present in the results and throws an `AssertionError` if not, providing a clear message for debugging. 

This implementation adheres to your requirements and uses the specified regex format for capturing parameters.
Here is a complete Java step definition implementation for the Cucumber step you provided:

```java
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import io.cucumber.java.en.Then;

public class SearchResultsSteps {
    private WebDriver driver;
    private WebDriverWait wait;

    public SearchResultsSteps(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, 10); // Wait for up to 10 seconds
    }

    @Then("^the results should include at least one item with \"([^\"]*)\" in the title$")
    public void theResultsShouldIncludeAtLeastOneItemWithInTheTitle(String keyword) {
        // Wait for the results container to be visible
        WebElement resultsContainer = wait.until(ExpectedConditions.visibilityOfElementLocated(By.id("results-container")));

        // Find all items in the results
        List<WebElement> items = resultsContainer.findElements(By.className("result-item"));

        // Check if at least one item contains the keyword in its title
        boolean itemFound = items.stream().anyMatch(item -> {
            String title = item.findElement(By.className("item-title")).getText();
            return title.toLowerCase().contains(keyword.toLowerCase());
        });

        // Assert that at least one item was found
        if (!itemFound) {
            throw new AssertionError("No items found with \"" + keyword + "\" in the title.");
        }
    }
}
```

### Explanation:
- The method `theResultsShouldIncludeAtLeastOneItemWithInTheTitle` takes a `keyword` parameter, which is captured using the regex pattern `"([^"]*)"`.
- It waits for the results container to be visible using `WebDriverWait`.
- It retrieves all items from the results and checks if any of them contain the specified keyword in their title.
- If no matching item is found, it throws an `AssertionError` to indicate the failure of the test.
Here is a complete Java step definition implementation for the Cucumber step "Given the search bar is visible":

```java
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import io.cucumber.java.en.Given;

public class SearchBarSteps {
    private WebDriver driver;
    private WebDriverWait wait;

    public SearchBarSteps(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, 10); // Adjust timeout as necessary
    }

    @Given("^the search bar is visible$")
    public void theSearchBarIsVisible() {
        WebElement searchBar = driver.findElement(By.id("search-bar")); // Replace with actual ID or locator
        wait.until(ExpectedConditions.visibilityOf(searchBar));
    }
}
```

### Explanation:
1. **WebDriver and WebDriverWait**: The `WebDriver` instance is used to interact with the browser, and `WebDriverWait` is used to wait for certain conditions (like visibility of elements).
2. **Step Definition**: The method `theSearchBarIsVisible` checks if the search bar is visible on the page by waiting until the element is visible.
3. **Locator**: The `By.id("search-bar")` is a placeholder for the actual locator of the search bar. You should replace `"search-bar"` with the actual ID or locator used in your application.
Here is the complete Java step definition implementation for the given Cucumber step:

```java
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import io.cucumber.java.en.When;

public class SearchStepDefinitions {
    private WebDriver driver;
    private WebDriverWait wait;

    public SearchStepDefinitions(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, 10); // Set timeout to 10 seconds
    }

    @When("^the user enters \"([^\"]*)\" in the search bar$")
    public void theUserEntersSearchTermInTheSearchBar(String searchTerm) {
        WebElement searchBar = driver.findElement(By.id("search-bar")); // Replace with actual search bar ID
        wait.until(ExpectedConditions.elementToBeClickable(searchBar));
        searchBar.clear();
        searchBar.sendKeys(searchTerm);
    }
}
```

### Explanation:
- The `@When` annotation is used to define the step in Cucumber.
- The regex pattern `^the user enters "([^"]*)" in the search bar$` captures the search term provided in quotes.
- The method `theUserEntersSearchTermInTheSearchBar` takes the captured parameter `searchTerm` and uses it to interact with the search bar.
- The `WebDriverWait` is used to ensure that the search bar is clickable before sending keys to it.
- The `By.id("search-bar")` should be replaced with the actual ID of the search bar in your application.
public void userClicksButton(String buttonText) {
WebElement button = wait.until(ExpectedConditions.elementToBeClickable(
            By.xpath("//button[contains(text(),'" + buttonText + "')]")));
        button.click();
}
Here's the complete Java step definition implementation for the Cucumber step you provided:

```java
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import io.cucumber.java.en.Then;

public class SearchResultsSteps {
    private WebDriver driver;
    private WebDriverWait wait;

    public SearchResultsSteps(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, 10); // Adjust timeout as needed
    }

    @Then("^the search results should display \"([^\"]*)\" message$")
    public void theSearchResultsShouldDisplayMessage(String expectedMessage) {
        // Locate the element that displays the search results message
        WebElement messageElement = wait.until(ExpectedConditions.visibilityOfElementLocated(By.id("search-results-message")));
        
        // Get the actual message text
        String actualMessage = messageElement.getText();
        
        // Assert that the actual message matches the expected message
        if (!actualMessage.equals(expectedMessage)) {
            throw new AssertionError("Expected message: \"" + expectedMessage + "\", but found: \"" + actualMessage + "\"");
        }
    }
}
```

### Explanation:
1. **Imports**: Necessary Selenium and Cucumber imports are included.
2. **Class Definition**: The class `SearchResultsSteps` contains the step definitions related to search results.
3. **Constructor**: A constructor initializes the `WebDriver` and `WebDriverWait` instances.
4. **Step Definition**: The method `theSearchResultsShouldDisplayMessage` captures the expected message using the regex pattern `"([^"]*)"` and checks if the displayed message matches the expected one.
5. **Element Location**: The message element is located using its ID (`search-results-message`), which you should replace with the actual ID used in your application.
6. **Assertion**: An assertion is performed to compare the actual message with the expected message, throwing an `AssertionError` if they do not match.
Here is a complete Java step definition implementation for the Cucumber step "Given the search bar is visible":

```java
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import io.cucumber.java.en.Given;

public class SearchBarStepDefinitions {

    private WebDriver driver;
    private WebDriverWait wait;

    public SearchBarStepDefinitions(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, 10); // 10 seconds wait time
    }

    @Given("^the search bar is visible$")
    public void theSearchBarIsVisible() {
        WebElement searchBar = driver.findElement(By.id("search-bar")); // Assuming the search bar has an ID of "search-bar"
        wait.until(ExpectedConditions.visibilityOf(searchBar));
        // Optionally, you can assert that the search bar is displayed
        if (!searchBar.isDisplayed()) {
            throw new AssertionError("Search bar is not visible");
        }
    }
}
```

### Explanation:
- The `@Given` annotation is used to define the step in Cucumber.
- The method `theSearchBarIsVisible` is implemented to check the visibility of the search bar.
- The `WebDriverWait` is used to wait until the search bar is visible on the page.
- An assertion is included to throw an error if the search bar is not displayed, ensuring that the step fails if the condition is not met.
Here is the complete Java step definition implementation for the Cucumber step "When the user leaves the search bar empty":

```java
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import io.cucumber.java.en.When;

public class SearchSteps {
    private WebDriver driver;
    private WebDriverWait wait;

    public SearchSteps(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, 10);
    }

    @When("^the user leaves the search bar empty$")
    public void userLeavesSearchBarEmpty() {
        WebElement searchBar = driver.findElement(By.id("search-bar"));
        wait.until(ExpectedConditions.elementToBeClickable(searchBar));
        searchBar.clear(); // Ensure the search bar is empty
        // No need to send keys as we are leaving it empty
    }
}
```

### Explanation:
- The `@When` annotation is used to define the step in Cucumber.
- The method `userLeavesSearchBarEmpty` is implemented to interact with the search bar.
- The `WebElement` for the search bar is located using its ID (`search-bar`).
- The `WebDriverWait` is used to ensure that the search bar is clickable before interacting with it.
- The `clear()` method is called to ensure that the search bar is empty, fulfilling the requirement of leaving it empty.
public void userClicksButton(String buttonText) {
WebElement button = wait.until(ExpectedConditions.elementToBeClickable(
            By.xpath("//button[contains(text(),'" + buttonText + "')]")));
        button.click();
}
Here is the complete Java step definition implementation for the specified Cucumber step:

```java
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import io.cucumber.java.en.Then;

public class SearchStepDefinitions {
    private WebDriver driver;
    private WebDriverWait wait;

    public SearchStepDefinitions(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, 10); // Adjust timeout as needed
    }

    @Then("^the search results should display \"([^\"]*)\" message$")
    public void theSearchResultsShouldDisplayMessage(String expectedMessage) {
        // Locate the element that displays the search message
        WebElement messageElement = wait.until(ExpectedConditions.visibilityOfElementLocated(By.id("search-message")));

        // Get the actual message text from the element
        String actualMessage = messageElement.getText();

        // Assert that the actual message matches the expected message
        if (!actualMessage.equals(expectedMessage)) {
            throw new AssertionError("Expected message: \"" + expectedMessage + "\", but got: \"" + actualMessage + "\"");
        }
    }
}
```

### Explanation:
1. **Imports**: Necessary Selenium and Cucumber imports are included.
2. **WebDriver and WebDriverWait**: The constructor initializes the WebDriver and WebDriverWait.
3. **Step Definition**: The method `theSearchResultsShouldDisplayMessage` captures the expected message using the regex pattern `\"([^\"]*)\"`.
4. **Element Location**: The message element is located using its ID (`search-message`). Adjust the ID based on your actual HTML structure.
5. **Assertion**: The method checks if the actual message matches the expected message and throws an `AssertionError` if they do not match.
Here is a complete Java step definition implementation for the Cucumber step "Given the search bar is visible":

```java
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import io.cucumber.java.en.Given;

public class SearchBarSteps {
    private WebDriver driver;
    private WebDriverWait wait;

    public SearchBarSteps(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, 10); // Adjust timeout as necessary
    }

    @Given("^the search bar is visible$")
    public void theSearchBarIsVisible() {
        WebElement searchBar = driver.findElement(By.id("search-bar")); // Replace with actual ID or locator
        wait.until(ExpectedConditions.visibilityOf(searchBar));
    }
}
```

### Explanation:
- The `@Given` annotation is used to define the step in Cucumber.
- The method `theSearchBarIsVisible` checks for the visibility of the search bar element.
- The `WebDriverWait` is used to wait until the search bar is visible, ensuring that the test does not proceed until the element is ready for interaction.
- Replace `"search-bar"` with the actual ID or locator of your search bar element as needed.
Here is the complete Java step definition implementation for the Cucumber step you provided:

```java
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import io.cucumber.java.en.When;

public class SearchStepDefinitions {
    private WebDriver driver;
    private WebDriverWait wait;

    public SearchStepDefinitions(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, 10); // Adjust timeout as needed
    }

    @When("^the user enters \"([^\"]*)\" in the search bar$")
    public void theUserEntersSpecialCharactersInSearchBar(String searchTerm) {
        WebElement searchBar = driver.findElement(By.id("search-bar")); // Adjust the locator as needed
        wait.until(ExpectedConditions.elementToBeClickable(searchBar));
        searchBar.clear();
        searchBar.sendKeys(searchTerm);
    }
}
```

### Explanation:
- The `@When` annotation specifies the Cucumber step and uses the regex pattern `"([^"]*)"` to capture the parameter.
- The method `theUserEntersSpecialCharactersInSearchBar` takes a `String` parameter `searchTerm`, which will be the special characters `!@#$%`.
- The `WebElement` for the search bar is located using `By.id("search-bar")`. You may need to adjust the locator based on your actual HTML structure.
- The `WebDriverWait` is used to ensure that the search bar is clickable before interacting with it.
- The search bar is cleared and the special characters are entered using `sendKeys(searchTerm)`.
public void userClicksButton(String buttonText) {
WebElement button = wait.until(ExpectedConditions.elementToBeClickable(
            By.xpath("//button[contains(text(),'" + buttonText + "')]")));
        button.click();
}
Here's the complete Java step definition implementation for the Cucumber step you provided:

```java
import org.openqa.selenium.By;
import org.openqa.selenium.WebDriver;
import org.openqa.selenium.WebElement;
import org.openqa.selenium.support.ui.ExpectedConditions;
import org.openqa.selenium.support.ui.WebDriverWait;
import io.cucumber.java.en.Then;

public class SearchResultsSteps {
    private WebDriver driver;
    private WebDriverWait wait;

    public SearchResultsSteps(WebDriver driver) {
        this.driver = driver;
        this.wait = new WebDriverWait(driver, 10); // 10 seconds wait time
    }

    @Then("^the search results should display \"([^\"]*)\" message$")
    public void theSearchResultsShouldDisplayMessage(String expectedMessage) {
        WebElement messageElement = wait.until(ExpectedConditions.visibilityOfElementLocated(By.id("no-results-message")));
        String actualMessage = messageElement.getText();
        if (!actualMessage.equals(expectedMessage)) {
            throw new AssertionError("Expected message: " + expectedMessage + " but found: " + actualMessage);
        }
    }
}
```

### Explanation:
1. **Imports**: Necessary classes from Selenium and Cucumber are imported.
2. **Constructor**: The `SearchResultsSteps` class constructor initializes the `WebDriver` and `WebDriverWait`.
3. **Step Definition**: The method `theSearchResultsShouldDisplayMessage` captures the expected message using the regex pattern `"([^"]*)"` and checks if the actual message displayed on the page matches the expected message.
4. **WebDriver Code**: It waits for the message element to be visible and retrieves its text to perform the assertion. If the messages do not match, an `AssertionError` is thrown with a descriptive message.

}